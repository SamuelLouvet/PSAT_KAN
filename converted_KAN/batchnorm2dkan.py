import torch
import torch.nn as nn
from torch.fx.proxy import Proxy


class AffineKAN(nn.Module):
    """
    Per-channel affine transform expressed as KAN-friendly operations.

    y = x * scale + bias
    """

    def __init__(self, scale: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.scale = nn.Parameter(scale)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, Proxy):
            if x.dim() != 4:
                raise ValueError("AffineKAN expects input of shape (N, C, H, W)")
            C = x.size(1)
            if C != self.scale.numel():
                raise ValueError(
                    f"Expected {self.scale.numel()} channels, got {C}"
                )
        else:
            C = self.scale.numel()
        scale = self.scale.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        return x * scale + bias


BatchNorm2dKAN = AffineKAN
