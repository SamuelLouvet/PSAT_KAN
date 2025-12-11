"""
Utilities to rewrite standard PyTorch modules into KAN-friendly equivalents
while copying weights/biases.
"""

import copy
import torch
import torch.nn as nn

from converted_KAN.conv2dkan import Conv2dKAN
from converted_KAN.linearkan import LinearKAN
from converted_KAN.avgpool2dkan import AvgPool2dKAN
from converted_KAN.relumaxpool2d import ReLUMaxPool2dKAN


def _clone_conv2d(module: nn.Conv2d) -> Conv2dKAN:
    if module.padding_mode != "zeros":
        raise NotImplementedError("Conv2dKAN only supports padding_mode='zeros'")

    new = Conv2dKAN(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    ).to(device=module.weight.device, dtype=module.weight.dtype)

    new.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        new.bias.data.copy_(module.bias.data)
    return new


def _clone_linear(module: nn.Linear) -> LinearKAN:
    new = LinearKAN(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
    ).to(device=module.weight.device, dtype=module.weight.dtype)

    new.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        new.bias.data.copy_(module.bias.data)
    return new


def _clone_avgpool2d(module: nn.AvgPool2d) -> AvgPool2dKAN:
    if module.ceil_mode:
        raise NotImplementedError("AvgPool2dKAN does not support ceil_mode=True")
    if module.divisor_override is not None:
        raise NotImplementedError("AvgPool2dKAN does not support divisor_override")

    return AvgPool2dKAN(
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        count_include_pad=module.count_include_pad,
    )


def _clone_maxpool2d(module: nn.MaxPool2d) -> ReLUMaxPool2dKAN:
    if module.ceil_mode:
        raise NotImplementedError("ReLUMaxPool2dKAN does not support ceil_mode=True")
    if module.return_indices:
        raise NotImplementedError("ReLUMaxPool2dKAN does not return indices")
    if getattr(module, "dilation", 1) != 1:
        raise NotImplementedError("ReLUMaxPool2dKAN assumes dilation=1")

    return ReLUMaxPool2dKAN(
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
    )


def _convert_leaf(module: nn.Module) -> nn.Module:
    if isinstance(module, Conv2dKAN):
        return module
    if isinstance(module, LinearKAN):
        return module
    if isinstance(module, AvgPool2dKAN):
        return module
    if isinstance(module, ReLUMaxPool2dKAN):
        return module

    if isinstance(module, nn.Conv2d):
        return _clone_conv2d(module)
    if isinstance(module, nn.Linear):
        return _clone_linear(module)
    if isinstance(module, nn.AvgPool2d):
        return _clone_avgpool2d(module)
    if isinstance(module, nn.MaxPool2d):
        return _clone_maxpool2d(module)

    return module


def convert_to_kan(module: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Recursively replaces supported layers in `module` with KAN variants.

    Supported:
      - nn.Conv2d -> Conv2dKAN (padding_mode must be 'zeros')
      - nn.Linear -> LinearKAN
      - nn.AvgPool2d -> AvgPool2dKAN (ceil_mode=False, count_include_pad=False)
      - nn.MaxPool2d -> ReLUMaxPool2dKAN (ceil_mode=False, dilation=1)
    """
    root = module if inplace else copy.deepcopy(module)

    def _convert(node: nn.Module) -> nn.Module:
        for name, child in list(node.named_children()):
            node._modules[name] = _convert(child)
        return _convert_leaf(node)

    return _convert(root)


# Alias for symmetry
to_kan = convert_to_kan
