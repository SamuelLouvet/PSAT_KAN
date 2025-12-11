import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def kan_pairwise_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise max with two explicit layers:
      - layer 1: linear mix to form a-b (and carry b)
      - layer 2: apply relu on (a-b) and sum with b

    max(a, b) = b + relu(a - b)
    """
    diff = a - b
    relu_diff = F.relu(diff)
    return b + relu_diff


def kan_max_reduce_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce last dim by repeated pairwise max using the two-step
    (a-b) -> relu -> b+relu construction.

    x: (..., n)
    Return: (...,)
    """
    while x.size(-1) > 1:
        n = x.size(-1)

        # For odd lengths, stash the last element
        if n % 2 == 1:
            last = x[..., -1:]
            x = x[..., :-1]
        else:
            last = None

        # Pair elements
        a = x[..., 0::2]
        b = x[..., 1::2]

        # Pairwise maximum via two-layer KAN construction
        x = kan_pairwise_max(a, b)

        # Append the saved element if needed
        if last is not None:
            x = torch.cat([x, last], dim=-1)

    # Now shape (..., 1)
    return x.squeeze(-1)


class ReLUMaxPool2dKAN(nn.Module):
    """
    MaxPool2d built from repeated pairwise maxima:
    layer1 (a-b), layer2 relu(a-b) + b (no F.max_pool2d).
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

        # 2) Reshape:
        # (B, C, L, kH*kW)
        n = kH * kW
        patches = patches.view(B, C, n, -1)
        patches = patches.permute(0, 1, 3, 2)

        # 3) Compute max over window dimension (n) via KAN pairwise max
        # Result: (B, C, L)
        y = kan_max_reduce_last_dim(patches)

        # 4) Reshape to target spatial size
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        y = y.view(B, C, H_out, W_out)
        return y


# Backward-compatible alias
ReLUMaxPool2d = ReLUMaxPool2dKAN
