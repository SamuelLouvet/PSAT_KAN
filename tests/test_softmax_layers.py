import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from converted_KAN.softmaxkan import SoftmaxKAN
from converted_KAN.ops_counter import count_ops


def test_softmax_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 7)

    std = nn.Softmax(dim=-1)
    kan = SoftmaxKAN(dim=-1, stable=True, eps=1e-8)

    with torch.no_grad():
        y_std = std(x)
        y_kan = kan(x)

    assert torch.allclose(y_std, y_kan, atol=1e-5, rtol=1e-4), (
        (y_std - y_kan).abs().max().item()
    )


def test_softmax_per_layer_fx_has_stages():
    model = SoftmaxKAN(dim=-1, stable=True, eps=1e-8)
    result = count_ops(model, input_shape=(1, 100), per_layer=True)

    per_layer = result.get("per_layer", {})
    assert len(per_layer) >= 3, per_layer
    keys = list(per_layer.keys())
    assert any("max_op" in k for k in keys), keys
    assert any("division" in k for k in keys), keys


def test_softmax_fx_max_is_kan_not_amax():
    model = SoftmaxKAN(dim=-1, stable=True, eps=1e-8)
    result = count_ops(model, input_shape=(1, 100), per_layer=False)
    total = result.get("total", {})
    assert total.get("compare", 0) == 0, total


def main():
    print("SoftmaxKAN layerwise tests")
    test_softmax_matches_torch()
    print("  softmax matches torch: PASSED")
    test_softmax_per_layer_fx_has_stages()
    print("  per-layer FX stages: PASSED")
    test_softmax_fx_max_is_kan_not_amax()
    print("  FX max uses KAN ops: PASSED")


if __name__ == "__main__":
    main()
