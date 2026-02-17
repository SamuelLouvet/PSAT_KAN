import torch
import torch.nn as nn

from .relumaxpool2d import MaxReduceLastDimFixedKAN, kan_max_reduce_last_dim


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


class _KANAddDiff(nn.Module):
    """
    Layer 1 for KAN multiply: compute (a+b) and (a-b).
    Returns a tuple (sum_ab, diff_ab).
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return a + b, a - b


class _KANApplySquarePair(nn.Module):
    """
    Layer 2 for KAN multiply: apply x^2 to both tensors in a pair.
    Input: (x1, x2) -> Output: (x1^2, x2^2)
    """

    def forward(
        self, pair: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = pair
        return kan_square(x1), kan_square(x2)


class _KANSubScaleQuarter(nn.Module):
    """
    Layer 3 for KAN multiply expressed as:
      term1 = 0.25 * sum_sq   (unary scaling)
      term2 = -0.25 * diff_sq (unary scaling)
      out = sum([term1, term2])  (sum reduction over 2 terms)

    This matches the "functions then sum" KAN pattern more closely.
    """

    def forward(self, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sum_sq, diff_sq = pair
        term1 = kan_scale_quarter(sum_sq)
        term2 = (-0.25) * diff_sq
        stacked = torch.stack((term1, term2), dim=0)
        return stacked.sum(dim=0)


class MultiplyKAN(nn.Module):
    """
    Multiplication expressed as an explicit 3-layer KAN composition:
      1) additions (a+b, a-b)
      2) unary square on both
      3) subtraction + unary scale by 0.25
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = _KANAddDiff()
        self.layer2 = _KANApplySquarePair()
        self.layer3 = _KANSubScaleQuarter()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(a, b)))


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


class ReciprocalKAN(nn.Module):
    """
    Unary reciprocal: 1/(x+eps).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kan_reciprocal(x, eps=self.eps)


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
    if isinstance(x, torch.Tensor):
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
    else:
        if dim != -1:
            raise ValueError("kan_max FX trace only supports dim=-1")

    reduced = kan_max_reduce_last_dim(x)
    return reduced.unsqueeze(-1) if keepdim else reduced


class MaxKAN(nn.Module):
    """
    Max reduction wrapped as a module so FX per-layer accounting can attribute ops.
    """

    def __init__(self, dim: int = -1, keepdim: bool = True, n: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.n: int | None = None
        self.fixed_reduce: nn.Module | None = None
        if n is not None:
            self.configure_fixed_n(n)

    def configure_fixed_n(self, n: int | None) -> None:
        """
        Configure a fixed-size, FX-traceable max reduction.

        This is useful for FX tracing / op counting, because the dynamic
        pairwise reduction uses data-dependent Python control flow.

        Currently supports only dim=-1.
        """
        if n is None:
            self.n = None
            self.fixed_reduce = None
            return

        self.n = int(n)
        self.fixed_reduce = MaxReduceLastDimFixedKAN(n=self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_reduce is not None:
            # FX tracing: avoid any Python control-flow depending on Proxy values.
            # MaxKAN tracing mode historically supports dim=-1 only.
            if not isinstance(x, torch.Tensor):
                reduced = self.fixed_reduce(x)
                return reduced.unsqueeze(-1) if self.keepdim else reduced

            dim = int(self.dim)
            if dim < 0:
                dim = x.dim() + dim
            if dim < 0 or dim >= x.dim():
                raise ValueError(f"dim out of range: {self.dim}")

            if dim != x.dim() - 1:
                perm = [d for d in range(x.dim()) if d != dim] + [dim]
                x = x.permute(*perm)
                reduced = self.fixed_reduce(x)
                if self.keepdim:
                    reduced = reduced.unsqueeze(-1)
                    inv_perm = [perm.index(i) for i in range(len(perm))]
                    reduced = reduced.permute(*inv_perm)
                return reduced

            reduced = self.fixed_reduce(x)
            return reduced.unsqueeze(-1) if self.keepdim else reduced
        return kan_max(x, dim=self.dim, keepdim=self.keepdim)


class ExpKAN(nn.Module):
    """
    Unary exp wrapped as a module (for per-layer FX accounting).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class SumKAN(nn.Module):
    """
    Sum reduction wrapped as a module (for per-layer FX accounting).
    """

    def __init__(self, dim: int = -1, keepdim: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim, keepdim=self.keepdim)


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
        self.reciprocal = ReciprocalKAN(eps=eps)
        self.multiply = MultiplyKAN()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute a/b using KAN operations.

        a: numerator tensor
        b: denominator tensor (must be positive for stability)
        """
        b_recip = self.reciprocal(b)
        return self.multiply(a, b_recip)


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
        self.max_op = MaxKAN(dim=dim, keepdim=True)
        self.exp_op = ExpKAN()
        self.sum_op = SumKAN(dim=dim, keepdim=True)
        self.division = DivisionKAN(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input tensor
        Returns: softmax along specified dimension
        """
        if self.stable:
            # Stable softmax: subtract max for numerical stability (KAN-style max)
            x_max = self.max_op(x)
            x_shifted = x - x_max
        else:
            x_shifted = x

        # Layer 1: exp (unary function)
        exp_x = self.exp_op(x_shifted)

        # Layer 2: sum (addition) + reciprocal preparation
        sum_exp = self.sum_op(exp_x)

        # Layer 3: division using KAN (exp_x / sum_exp)
        return self.division(exp_x, sum_exp)


# Backward-compatible alias
Softmax = SoftmaxKAN
Division = DivisionKAN
