import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class AvgPool2dKAN(nn.Module):
    """
    AvgPool2d built from unfold + linear operations
    (KAN-compatible, no F.avg_pool2d).
    """

    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()

        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride if stride is not None else kernel_size)
        self.padding = _to_2tuple(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        # 1) Extract patches:
        # (B, C*kH*kW, L)
        patches = F.unfold(
            x,
            kernel_size=(kH, kW),
            padding=(pH, pW),
            stride=(sH, sW)
        )

        # 2) Reshape to separate channels and patch elements:
        # (B, C, L, kH*kW)
        n = kH * kW
        patches = patches.view(B, C, n, -1)
        patches = patches.permute(0, 1, 3, 2)

        # 3) Sum over window dimension (n) then scale by 1/n
        # Result: (B, C, L)
        y = patches.sum(dim=-1) * (1.0 / n)

        # 4) Reshape to target spatial size
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        y = y.view(B, C, H_out, W_out)
        return y
