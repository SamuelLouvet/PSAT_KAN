from typing import Iterable

import torch
import torch.nn as nn


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
