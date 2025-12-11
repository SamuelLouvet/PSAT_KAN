import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from converted_KAN.conv2dkan import Conv2dKAN

def test_conv2d():
    print("Conv2dKAN / nn.Conv2d")
    
    configs = [
        {'in_channels': 3, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1},
        {'in_channels': 3, 'out_channels': 8, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1},
    ]

    for i, config in enumerate(configs):
        print(f"  Config {i+1}: {config}")
        
        std_conv = nn.Conv2d(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['padding'],
            dilation=config['dilation'],
            bias=True
        )
        
        custom_conv = Conv2dKAN(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['padding'],
            dilation=config['dilation'],
            bias=True
        )
        
        with torch.no_grad():
            custom_conv.weight.copy_(std_conv.weight)
            custom_conv.bias.copy_(std_conv.bias)
            
        x = torch.randn(2, config['in_channels'], 32, 32)
        
        with torch.no_grad():
            std_out = std_conv(x)
            custom_out = custom_conv(x)
            
        try:
            assert torch.allclose(std_out, custom_out, atol=1e-5)
            print(f"    Max diff: {(std_out - custom_out).abs().max().item()}")
            print("    -> PASSED")
        except AssertionError:
            print("    -> FAILED")
            print(f"    Max diff: {(std_out - custom_out).abs().max().item()}")
            return

    print("\nAll Conv2d tests passed!")

if __name__ == "__main__":
    test_conv2d()
