import math
import torch
import torch.nn as nn
from torch.fx.proxy import Proxy

from .softmaxkan import MultiplyKAN, SumKAN


class LinearKAN(nn.Module):
    """
    Linear layer expressed via explicit matmul (KAN-friendly).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.multiply = MultiplyKAN()
        self.sum_in = SumKAN(dim=-1, keepdim=False)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        """
        if not isinstance(x, Proxy):
            if x.size(-1) != self.in_features:
                raise ValueError(
                    f"Expected input features {self.in_features}, got {x.size(-1)}"
                )

        # Same idea as a linear layer: compute products and sum them
        x_exp = x.unsqueeze(-2)
        out = self.sum_in(self.multiply(x_exp, self.weight))
        if self.bias is not None:
            out = out + self.bias
        return out


# Backward-compatible alias
Linear = LinearKAN
