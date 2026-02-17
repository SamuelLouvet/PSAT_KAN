import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from converted_KAN.relumaxpool2d import ReLUMaxPool2dKAN
from converted_KAN.ops_counter import count_ops


def test_maxpool_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 8)

    std = nn.MaxPool2d(kernel_size=2, stride=2)
    kan = ReLUMaxPool2dKAN(kernel_size=2, stride=2)

    with torch.no_grad():
        y_std = std(x)
        y_kan = kan(x)

    assert torch.allclose(y_std, y_kan, atol=1e-5, rtol=1e-4), (
        (y_std - y_kan).abs().max().item()
    )


def test_maxpool_per_layer_fx_has_pairwise_layers():
    model = ReLUMaxPool2dKAN(kernel_size=2, stride=2)
    result = count_ops(model, input_shape=(1, 3, 8, 8), per_layer=True)

    per_layer = result.get("per_layer", {})
    keys = list(per_layer.keys())
    assert len(per_layer) >= 2, keys
    assert any("window_max" in k for k in keys), keys
    assert any("pairwise_max" in k for k in keys), keys


def main():
    print("ReLUMaxPool2dKAN layerwise tests")
    test_maxpool_matches_torch()
    print("  maxpool matches torch: PASSED")
    test_maxpool_per_layer_fx_has_pairwise_layers()
    print("  per-layer FX stages: PASSED")


if __name__ == "__main__":
    main()

