import copy
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from .convert import convert_to_kan
from .ops_counter import count_ops
from .quantization import quantize_model


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader: Iterable,
    *,
    device: str = "cpu",
    show_progress: bool = False,
) -> float:
    """
    Evaluate top-1 accuracy for classification logits.
    """
    model = model.to(device=device).eval()
    total = 0
    correct = 0
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc="eval", leave=False)
        except Exception:
            iterator = dataloader
    else:
        iterator = dataloader

    for batch in iterator:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError("Dataloader must yield (inputs, targets)")
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        total += targets.numel()
        correct += (preds == targets).sum().item()
    if total == 0:
        return 0.0
    return correct / total


def accuracy_cost_sweep(
    model: nn.Module,
    dataloader: Iterable,
    input_shape: Tuple[int, ...],
    bit_widths: Iterable[int],
    *,
    device: str = "cpu",
    convert_kan: bool = True,
    quantize_buffers: bool = True,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> List[Dict[str, Any]]:
    """
    Sweep quantization bit-widths and report accuracy + op cost.
    """
    base_model = convert_to_kan(model, inplace=False) if convert_kan else copy.deepcopy(model)
    results: List[Dict[str, Any]] = []

    for bits in bit_widths:
        q_model = quantize_model(
            base_model,
            num_bits=bits,
            per_channel=per_channel,
            channel_dim=channel_dim,
            quantize_buffers=quantize_buffers,
            inplace=False,
        )
        acc = evaluate_accuracy(q_model, dataloader, device=device)
        ops = count_ops(q_model, input_shape, device=device)["total"]
        results.append(
            {
                "bits": int(bits),
                "accuracy": float(acc),
                "ops": ops,
                "total_arith": int(ops.get("total_arith", 0)),
            }
        )
    return results
