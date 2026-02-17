import math
import torch
import torch.nn as nn
from torch.fx.proxy import Proxy

from .softmaxkan import (
    kan_square,
    kan_scale_quarter,
    kan_multiply,
    MultiplyKAN,
    SumKAN,
    SoftmaxKAN,
)
from .linearkan import LinearKAN


def kan_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Unary function: x * scale (linear scaling)
    """
    return x * scale


class ScaleKAN(nn.Module):
    """
    Unary scaling wrapped as a module (for per-layer FX accounting).
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kan_scale(x, self.scale)


class MatMulKAN(nn.Module):
    """
    KAN-style matmul as explicit stages:
      - elementwise MultiplyKAN on broadcasted pairs
      - SumKAN over the K dimension
    """

    def __init__(self):
        super().__init__()
        self.multiply = MultiplyKAN()
        self.sum_k = SumKAN(dim=-2, keepdim=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (..., M, K)
        # b: (..., K, N)
        a_expanded = a.unsqueeze(-1)  # (..., M, K, 1)
        b_expanded = b.unsqueeze(-3)  # (..., 1, K, N)
        products = self.multiply(a_expanded, b_expanded)  # (..., M, K, N)
        return self.sum_k(products)  # (..., M, N)


def kan_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    KAN-style matrix multiplication using element-wise KAN multiply and sum.

    For matrices A (... x M x K) and B (... x K x N):
    C_ij = Σ_k A_ik * B_kj

    Each product A_ik * B_kj is computed via KAN multiply:
    4ab = (a+b)² - (a-b)²

    The sum over k is a standard addition (allowed in KAN).

    Layer 1: For each (i,j,k), compute A_ik + B_kj and A_ik - B_kj
    Layer 2: Apply x² to each
    Layer 3: Subtract, scale by 0.25, sum over k
    """
    # a: (..., M, K)
    # b: (..., K, N)
    # We need to compute sum over k of a[..., i, k] * b[..., k, j]

    # Expand dimensions for broadcasting:
    # a: (..., M, K, 1)
    # b: (..., 1, K, N)
    a_expanded = a.unsqueeze(-1)  # (..., M, K, 1)
    b_expanded = b.unsqueeze(-3)  # (..., 1, K, N)

    # KAN multiply for each element pair
    # Layer 1: additions
    sum_ab = a_expanded + b_expanded  # (..., M, K, N)
    diff_ab = a_expanded - b_expanded  # (..., M, K, N)

    # Layer 2: unary function x²
    sum_sq = kan_square(sum_ab)
    diff_sq = kan_square(diff_ab)

    # Layer 3: subtract, scale
    products = kan_scale_quarter(sum_sq - diff_sq)  # (..., M, K, N)

    # Sum over K dimension (addition, allowed in KAN)
    return products.sum(dim=-2)  # (..., M, N)


def kan_batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    KAN-style batched matrix multiplication for attention.

    a: (B, H, S, D) - queries or attention weights
    b: (B, H, D, S) or (B, H, S, D) - keys transposed or values

    Returns: (B, H, S, S) or (B, H, S, D)
    """
    return kan_matmul(a, b)


class AttentionKAN(nn.Module):
    """
    Multi-Head Attention expressed entirely as KAN operations.

    Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V

    KAN decomposition:
    1. Q, K, V projections: Linear layers (can use LinearKAN)
    2. Q @ K^T: Matrix multiplication via KAN
    3. Scale by 1/sqrt(d_k): Unary scaling function
    4. Softmax: SoftmaxKAN
    5. scores @ V: Matrix multiplication via KAN

    All multiplications use: 4ab = (a+b)² - (a-b)²
    All divisions use: a/b = a × (1/b) with reciprocal as unary function
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Scale factor: 1/sqrt(d_k) as a constant for unary scaling
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.scale_op = ScaleKAN(self.scale)

        # Q, K, V projections using KAN linear layers
        self.q_proj = LinearKAN(embed_dim, embed_dim, bias=bias)
        self.k_proj = LinearKAN(embed_dim, embed_dim, bias=bias)
        self.v_proj = LinearKAN(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = LinearKAN(embed_dim, embed_dim, bias=bias)

        # KAN Softmax
        self.softmax = SoftmaxKAN(dim=-1, eps=eps)

        # Dropout (optional, not part of KAN but useful for training)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # KAN matmul stages
        self.matmul = MatMulKAN()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, S_q, E) where S_q is query sequence length
            key: (B, S_k, E) where S_k is key sequence length
            value: (B, S_k, E)
            attn_mask: (S_q, S_k) or (B, S_q, S_k), optional attention mask

        Returns:
            output: (B, S_q, E)
        """
        B, S_q, E = query.shape
        S_k = key.shape[1]

        # Step 1: Q, K, V projections (linear layers = unary functions f(x) = wx + b)
        Q = self.q_proj(query)  # (B, S_q, E)
        K = self.k_proj(key)  # (B, S_k, E)
        V = self.v_proj(value)  # (B, S_k, E)

        # Reshape for multi-head attention
        # (B, S, E) -> (B, S, H, D) -> (B, H, S, D)
        Q = Q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 2: Q @ K^T using KAN matrix multiplication
        # Q: (B, H, S_q, D), K^T: (B, H, D, S_k)
        K_t = K.transpose(-2, -1)  # (B, H, D, S_k)
        attn_scores = self.matmul(Q, K_t)  # (B, H, S_q, S_k)

        # Step 3: Scale by 1/sqrt(d_k) - unary scaling function
        attn_scores = self.scale_op(attn_scores)

        # Apply attention mask if provided
        # Note: FX tracing passes optional args as Proxy placeholders; avoid
        # Python control-flow on Proxy values for traceability in ops counting.
        if attn_mask is not None and not isinstance(attn_mask, Proxy):
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores + attn_mask

        # Step 4: Softmax using KAN
        attn_weights = self.softmax(attn_scores)  # (B, H, S_q, S_k)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Step 5: attn_weights @ V using KAN matrix multiplication
        # attn_weights: (B, H, S_q, S_k), V: (B, H, S_k, D)
        output = self.matmul(attn_weights, V)  # (B, H, S_q, D)

        # Reshape back: (B, H, S_q, D) -> (B, S_q, H, D) -> (B, S_q, E)
        output = output.transpose(1, 2).contiguous().view(B, S_q, E)

        # Output projection
        output = self.out_proj(output)

        return output


class SelfAttentionKAN(AttentionKAN):
    """
    Self-Attention layer where Q, K, V all come from the same input.
    """

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, S, E) input tensor
            attn_mask: optional attention mask

        Returns:
            output: (B, S, E)
        """
        return super().forward(x, x, x, attn_mask)


# Backward-compatible aliases
Attention = AttentionKAN
SelfAttention = SelfAttentionKAN
