# file: lowlevel_conv2d.py

from typing import Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


class Conv2dKAN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        kh, kw = self.kernel_size

        # (out_channels, in_channels, kh, kw)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kh, kw)
        )

        #  (out_channels,)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C_in, H, W)
        Retour: (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        assert C_in == self.in_channels, "Nombre de canaux d'entrée incorrect."

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # transforme chaque patch (C_in * kh * kw) en colonne
        # x_unfold: (N, C_in * kh * kw, L) où L = H_out * W_out
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

        # weight_flat: (C_out, C_in * kh * kw)
        weight_flat = self.weight.view(self.out_channels, -1)

        # x_unfold: (N, C_in*kh*kw, L)
        # weight_flat: (C_out, C_in*kh*kw)
        out = torch.matmul(weight_flat.unsqueeze(0), x_unfold)  # (N, C_out, L)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        out = out.view(N, self.out_channels, H_out, W_out)
        return out



