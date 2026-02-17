import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from converted_KAN.attentionkan import AttentionKAN
from converted_KAN.ops_counter import count_ops


def test_attention_matches_multiheadattention():
    torch.manual_seed(0)

    embed_dim = 32
    num_heads = 4
    seq_len = 8
    batch_size = 2

    attn_kan = AttentionKAN(embed_dim, num_heads, dropout=0.0)
    attn_std = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)

    with torch.no_grad():
        attn_std.in_proj_weight.copy_(
            torch.cat([attn_kan.q_proj.weight, attn_kan.k_proj.weight, attn_kan.v_proj.weight], dim=0)
        )
        attn_std.in_proj_bias.copy_(
            torch.cat([attn_kan.q_proj.bias, attn_kan.k_proj.bias, attn_kan.v_proj.bias], dim=0)
        )
        attn_std.out_proj.weight.copy_(attn_kan.out_proj.weight)
        attn_std.out_proj.bias.copy_(attn_kan.out_proj.bias)

    x = torch.randn(batch_size, seq_len, embed_dim)

    with torch.no_grad():
        y_kan = attn_kan(x, x, x)
        y_std, _ = attn_std(x, x, x, need_weights=False)

    assert torch.allclose(y_kan, y_std, atol=1e-4, rtol=1e-4), (
        (y_kan - y_std).abs().max().item()
    )


def test_attention_per_layer_fx_has_stages():
    embed_dim = 32
    num_heads = 4
    model = AttentionKAN(embed_dim, num_heads, dropout=0.0)

    class _Wrapper(nn.Module):
        def __init__(self, attn: nn.Module):
            super().__init__()
            self.attn = attn

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attn(x, x, x)

    result = count_ops(_Wrapper(model), input_shape=(2, 8, embed_dim), per_layer=True)
    per_layer = result.get("per_layer", {})
    keys = list(per_layer.keys())

    assert any("matmul.multiply.layer1" in k for k in keys), keys
    assert any("scale_op" in k for k in keys), keys
    assert any("softmax" in k for k in keys), keys


def main():
    print("AttentionKAN layerwise tests")
    test_attention_matches_multiheadattention()
    print("  matches MultiheadAttention: PASSED")
    test_attention_per_layer_fx_has_stages()
    print("  per-layer FX stages: PASSED")


if __name__ == "__main__":
    main()
