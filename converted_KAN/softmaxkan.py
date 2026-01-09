import torch
import torch.nn as nn

from .relumaxpool2d import kan_max_reduce_last_dim


def kan_square(x: torch.Tensor) -> torch.Tensor:
    """
    Unary function: x²
    """
    return x ** 2


def kan_scale_quarter(x: torch.Tensor) -> torch.Tensor:
    """
    Unary function: x * 0.25 (equivalent to x/4, but expressed as scaling)
    """
    return x * 0.25


def kan_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    KAN-style multiplication using the identity:
    4ab = (a+b)² - (a-b)²
    => ab = ((a+b)² - (a-b)²) * 0.25

    This converts multiplication into additions and unary functions (square, scale).

    Layer 1: Compute (a+b) and (a-b) [additions]
    Layer 2: Apply x² to each [unary function]
    Layer 3: Subtract and scale by 0.25 [addition + unary function]
    """
    # Layer 1: additions
    sum_ab = a + b
    diff_ab = a - b

    # Layer 2: unary function x²
    sum_sq = kan_square(sum_ab)
    diff_sq = kan_square(diff_ab)

    # Layer 3: subtraction (addition) + unary scaling
    diff_of_squares = sum_sq - diff_sq
    return kan_scale_quarter(diff_of_squares)


def kan_reciprocal(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Unary function: 1/x (reciprocal)

    Used only on positive domain to avoid discontinuity.
    eps is added for numerical stability.
    """
    return torch.reciprocal(x + eps)


def kan_division(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KAN-style division: a/b = a × (1/b)

    Converts division into:
    Layer 1: Compute 1/b (reciprocal, unary function)
    Layer 2: Compute a + (1/b) and a - (1/b) [additions]
    Layer 3: Apply x², subtract, scale by 0.25: ((a + 1/b)² - (a - 1/b)²) * 0.25

    See Figure 4 in the paper.
    """
    b_recip = kan_reciprocal(b, eps)
    return kan_multiply(a, b_recip)


def kan_max(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """
    KAN-style max reduction using pairwise max on the selected dimension.
    """
    if dim < 0:
        dim = x.dim() + dim
    if dim < 0 or dim >= x.dim():
        raise ValueError(f"dim out of range: {dim}")

    if dim != x.dim() - 1:
        perm = [d for d in range(x.dim()) if d != dim] + [dim]
        x = x.permute(*perm)
        reduced = kan_max_reduce_last_dim(x)
        if keepdim:
            reduced = reduced.unsqueeze(-1)
            inv_perm = [perm.index(i) for i in range(len(perm))]
            reduced = reduced.permute(*inv_perm)
        return reduced

    reduced = kan_max_reduce_last_dim(x)
    return reduced.unsqueeze(-1) if keepdim else reduced


class DivisionKAN(nn.Module):
    """
    Division layer expressed as KAN operations.

    a/b = a × (1/b) where:
    - 1/b is a unary reciprocal function
    - multiplication uses: 4ab = (a+b)² - (a-b)²
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute a/b using KAN operations.

        a: numerator tensor
        b: denominator tensor (must be positive for stability)
        """
        return kan_division(a, b, self.eps)


class SoftmaxKAN(nn.Module):
    """
    Softmax layer expressed as KAN operations.

    softmax(x)_i = exp(x_i) / Σ exp(x_j)

    KAN decomposition (3 layers):
    Layer 1: Apply exp(x) to each element (unary function)
    Layer 2: Sum all exp values, compute reciprocal 1/Σexp
    Layer 3: Multiply each exp(x_i) with 1/Σexp using KAN multiplication

    For numerical stability, uses stable softmax:
    softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    """

    def __init__(self, dim: int = -1, eps: float = 1e-8, stable: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.stable = stable
        self.division = DivisionKAN(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input tensor
        Returns: softmax along specified dimension
        """
        if self.stable:
            # Stable softmax: subtract max for numerical stability (KAN-style max)
            x_max = kan_max(x, dim=self.dim, keepdim=True)
            x_shifted = x - x_max
        else:
            x_shifted = x

        # Layer 1: exp (unary function)
        exp_x = torch.exp(x_shifted)

        # Layer 2: sum (addition) + reciprocal preparation
        sum_exp = exp_x.sum(dim=self.dim, keepdim=True)

        # Layer 3: division using KAN (exp_x / sum_exp)
        return self.division(exp_x, sum_exp)


# Backward-compatible alias
Softmax = SoftmaxKAN
Division = DivisionKAN
