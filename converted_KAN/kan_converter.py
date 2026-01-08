import torch
import torch.nn as nn
from .conv2dkan import Conv2dKAN
from .avgpool2dkan import AvgPool2dKAN
from .relumaxpool2d import ReLUMaxPool2dKAN
from .linearkan import LinearKAN

class KanConverter:
    def __init__(self):
        self.replacement_map = {
            nn.Conv2d: self._convert_conv2d,
            nn.AvgPool2d: self._convert_avgpool2d,
            nn.MaxPool2d: self._convert_maxpool2d,
            nn.Linear: self._convert_linear,
        }

    def convert_model(self, model: nn.Module) -> nn.Module:
        self._replace_layers_recursive(model)
        print(model)
        return model

    def _replace_layers_recursive(self, module: nn.Module):
        for name, child in module.named_children():
            child_type = type(child)
            
            print(f"Propcessing layer: {name} ({child_type.__name__})")

            if child_type in self.replacement_map:
                print(f"  -> Converting {child_type.__name__} to KAN version...")
                converter_func = self.replacement_map[child_type]
                new_layer = converter_func(child)
                setattr(module, name, new_layer)
            else:
                self._replace_layers_recursive(child)

    def _convert_conv2d(self, layer: nn.Conv2d) -> Conv2dKAN:
        kan_layer = Conv2dKAN(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=(layer.bias is not None)
        )
        
        with torch.no_grad():
            kan_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                kan_layer.bias.data = layer.bias.data.clone()
        
        return kan_layer

    def _convert_avgpool2d(self, layer: nn.AvgPool2d) -> AvgPool2dKAN:
        kan_layer = AvgPool2dKAN(
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            count_include_pad=layer.count_include_pad
        )
        return kan_layer

    def _convert_maxpool2d(self, layer: nn.MaxPool2d) -> ReLUMaxPool2dKAN:
        # Note: ReLUMaxPool2dKAN signature: kernel_size, stride=None, padding=0
        # ReLUMaxPool2dKAN might not support dilation, return_indices, ceil_mode
        kan_layer = ReLUMaxPool2dKAN(
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding
        )
        return kan_layer

    def _convert_linear(self, layer: nn.Linear) -> LinearKAN:
        kan_layer = LinearKAN(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=(layer.bias is not None)
        )
        
        # Copy weights
        with torch.no_grad():
            kan_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                kan_layer.bias.data = layer.bias.data.clone()
        
        return kan_layer
