import copy
import torch.nn as nn

from .convert import convert_to_kan
from .conv2dkan import Conv2dKAN
from .linearkan import LinearKAN
from .int_layers import Conv2dKANInt, LinearKANInt, ReLUInt
from .int_ops import dequantize_from_int, quantize_to_int


def _convert_leaf_int(module: nn.Module, frac_bits: int) -> nn.Module:
    if isinstance(module, (LinearKANInt, Conv2dKANInt, ReLUInt)):
        return module

    if isinstance(module, nn.ReLU):
        return ReLUInt()
    if isinstance(module, (LinearKAN, nn.Linear)):
        return LinearKANInt.from_float(module, frac_bits)
    if isinstance(module, (Conv2dKAN, nn.Conv2d)):
        return Conv2dKANInt.from_float(module, frac_bits)

    return module


def convert_to_int_kan(
    model: nn.Module,
    *,
    frac_bits: int = 8,
    inplace: bool = False,
    already_kan: bool = False,
) -> nn.Module:
    """
    Convert a float model to an integer-only KAN model (fixed-point).
    """
    root = model if inplace else copy.deepcopy(model)
    if not already_kan:
        root = convert_to_kan(root, inplace=True)

    def _convert(node: nn.Module) -> nn.Module:
        for name, child in list(node.named_children()):
            node._modules[name] = _convert(child)
        return _convert_leaf_int(node, frac_bits)

    return _convert(root)


class IntKANWrapper(nn.Module):
    """
    Wrap an integer KAN model: quantize inputs and optionally dequantize outputs.
    """

    def __init__(self, model_int: nn.Module, *, frac_bits: int = 8, return_int: bool = True):
        super().__init__()
        self.model_int = model_int
        self.frac_bits = frac_bits
        self.return_int = return_int

    def forward(self, x):
        qx = quantize_to_int(x, self.frac_bits)
        out = self.model_int(qx)
        if self.return_int:
            return out
        return dequantize_from_int(out, self.frac_bits)
