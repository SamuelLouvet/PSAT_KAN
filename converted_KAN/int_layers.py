from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .int_ops import INT32_MAX, INT32_MIN, _clamp_int32, kan_multiply_int, quantize_to_int


def _to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class ReLUInt(nn.Module):
    """
    Integer ReLU for fixed-point tensors.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0, INT32_MAX)


class LinearKANInt(nn.Module):
    """
    Integer-only KAN Linear with fixed-point arithmetic.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        frac_bits: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.frac_bits = frac_bits

        self.register_buffer("weight_int", torch.zeros(out_features, in_features, dtype=torch.int32))
        if bias:
            self.register_buffer("bias_int", torch.zeros(out_features, dtype=torch.int32))
        else:
            self.bias_int = None

    @classmethod
    def from_float(cls, layer: nn.Module, frac_bits: int = 8) -> "LinearKANInt":
        new = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            frac_bits=frac_bits,
            bias=layer.bias is not None,
        )
        new.weight_int = quantize_to_int(layer.weight.data, frac_bits)
        if layer.bias is not None:
            new.bias_int = quantize_to_int(layer.bias.data, frac_bits)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.int32:
            raise TypeError("LinearKANInt expects int32 input")
        if x.dim() != 2:
            raise ValueError("LinearKANInt expects input shape (B, in_features)")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input with {self.in_features} features, got {x.shape[1]}"
            )

        x_exp = x.unsqueeze(1)  # (B, 1, in_features)
        prod = kan_multiply_int(x_exp, self.weight_int, self.frac_bits)  # (B, out, in)
        out = prod.to(torch.int64).sum(dim=-1)
        if self.bias_int is not None:
            out = out + self.bias_int.to(torch.int64)
        return _clamp_int32(out).to(torch.int32)


class Conv2dKANInt(nn.Module):
    """
    Integer-only KAN Conv2d with fixed-point arithmetic (unfold + KAN multiply).
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
        *,
        frac_bits: int = 8,
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
        self.frac_bits = frac_bits

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        weight_shape = (
            out_channels,
            in_channels // groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        self.register_buffer("weight_int", torch.zeros(*weight_shape, dtype=torch.int32))
        if bias:
            self.register_buffer("bias_int", torch.zeros(out_channels, dtype=torch.int32))
        else:
            self.bias_int = None

    @classmethod
    def from_float(cls, layer: nn.Module, frac_bits: int = 8) -> "Conv2dKANInt":
        new = cls(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            frac_bits=frac_bits,
            bias=layer.bias is not None,
        )
        new.weight_int = quantize_to_int(layer.weight.data, frac_bits)
        if layer.bias is not None:
            new.bias_int = quantize_to_int(layer.bias.data, frac_bits)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.int32:
            raise TypeError("Conv2dKANInt expects int32 input")
        if x.dim() != 4:
            raise ValueError("Conv2dKANInt expects input of shape (B, C, H, W)")
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {C}")

        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

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

        patches = patches.view(B, self.groups, cin_g * kH * kW, L)
        weight = self.weight_int.view(self.groups, cout_g, cin_g * kH * kW)

        patches_exp = patches.unsqueeze(2)  # (B, g, 1, K, L)
        weight_exp = weight.unsqueeze(0).unsqueeze(-1)  # (1, g, O, K, 1)
        out = kan_multiply_int(patches_exp, weight_exp, self.frac_bits).sum(dim=-2)

        if self.bias_int is not None:
            out = out + self.bias_int.view(self.groups, cout_g).unsqueeze(0).unsqueeze(-1)

        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out = out.reshape(B, self.out_channels, H_out, W_out)
        return _clamp_int32(out.to(torch.int64)).to(torch.int32)
