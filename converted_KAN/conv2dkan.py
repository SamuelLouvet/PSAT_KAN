import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class Conv2dKAN(nn.Module):
    """
    Conv2d implemented via unfold + batched matmul
    (KAN-compatible; avoids torch.nn.functional.conv2d).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (
                (self.in_channels // self.groups)
                * self.kernel_size[0]
                * self.kernel_size[1]
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        """
        if x.dim() != 4:
            raise ValueError("Conv2dKAN expects input of shape (B, C, H, W)")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, got {C}"
            )

        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        # 1) Extract sliding local blocks
        patches = F.unfold(
            x,
            kernel_size=(kH, kW),
            dilation=(dH, dW),
            padding=(pH, pW),
            stride=(sH, sW),
        )  # (B, C_in*kH*kW, L)

        L = patches.shape[-1]
        cin_g = self.in_channels // self.groups
        cout_g = self.out_channels // self.groups

        # 2) Reshape for grouped matmul
        patches = patches.view(B, self.groups, cin_g * kH * kW, L)
        weight = self.weight.view(self.groups, cout_g, cin_g * kH * kW)

        # 3) Batched matmul over each group
        out = torch.einsum("bgil,goi->bgol", patches, weight)

        if self.bias is not None:
            out = out + self.bias.view(self.groups, cout_g).unsqueeze(0).unsqueeze(-1)

        # 4) Restore spatial layout
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out = out.reshape(B, self.out_channels, H_out, W_out)
        return out


# Backward-compatible alias
Conv2d = Conv2dKAN
