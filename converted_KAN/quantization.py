import copy
from typing import Iterable, Optional

import torch
import torch.nn as nn


def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int = 8,
    *,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> torch.Tensor:
    """
    Uniformly quantize a tensor and return de-quantized float values.
    """
    if num_bits < 2:
        raise ValueError("num_bits must be >= 2")
    if not torch.is_floating_point(tensor):
        return tensor

    if per_channel:
        t = tensor.transpose(0, channel_dim).contiguous()
        flat = t.reshape(t.shape[0], -1)
        mins = flat.min(dim=1).values
        maxs = flat.max(dim=1).values
        if symmetric:
            max_abs = torch.maximum(mins.abs(), maxs.abs())
            scale = max_abs / (2 ** (num_bits - 1) - 1)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.round(flat / scale[:, None])
            q = torch.clamp(q, -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
            dequant = q * scale[:, None]
        else:
            scale = (maxs - mins) / (2**num_bits - 1)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero_point = torch.round(-mins / scale)
            q = torch.round(flat / scale[:, None] + zero_point[:, None])
            q = torch.clamp(q, 0, 2**num_bits - 1)
            dequant = (q - zero_point[:, None]) * scale[:, None]
        dequant = dequant.reshape(t.shape).transpose(0, channel_dim).contiguous()
        return dequant

    t_min = tensor.min()
    t_max = tensor.max()
    if symmetric:
        max_abs = torch.maximum(t_min.abs(), t_max.abs())
        scale = max_abs / (2 ** (num_bits - 1) - 1)
        if scale == 0:
            return torch.zeros_like(tensor)
        q = torch.round(tensor / scale)
        q = torch.clamp(q, -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
        return q * scale

    scale = (t_max - t_min) / (2**num_bits - 1)
    if scale == 0:
        return torch.zeros_like(tensor)
    zero_point = torch.round(-t_min / scale)
    q = torch.round(tensor / scale + zero_point)
    q = torch.clamp(q, 0, 2**num_bits - 1)
    return (q - zero_point) * scale


def quantize_model(
    model: nn.Module,
    num_bits: int = 8,
    *,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = 0,
    inplace: bool = False,
    quantize_buffers: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> nn.Module:
    """
    Quantize parameters (and optionally buffers) of a model.
    """
    target = model if inplace else copy.deepcopy(model)
    skip = set(skip_names or [])

    for name, param in target.named_parameters():
        if name in skip or not torch.is_floating_point(param.data):
            continue
        param.data = quantize_tensor(
            param.data,
            num_bits,
            symmetric=symmetric,
            per_channel=per_channel,
            channel_dim=channel_dim,
        )

    if quantize_buffers:
        for name, buf in target.named_buffers():
            if name in skip or not torch.is_floating_point(buf):
                continue
            target._buffers[name] = quantize_tensor(
                buf,
                num_bits,
                symmetric=symmetric,
                per_channel=per_channel,
                channel_dim=channel_dim,
            )

    return target
