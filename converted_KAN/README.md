# PSAT_KAN
KAN-friendly PyTorch layers implemented with unfold + matmul/ReLU primitives:
- `AvgPool2dKAN`: average pooling via unfold + mean
 - `ReLUMaxPool2dKAN`: max pooling via two-step pairwise maxima `b + relu(a - b)` (no F.max_pool2d)
- `Conv2dKAN`: 2D convolution via unfold + grouped matmul
- `LinearKAN`: linear transform via explicit matmul
- `convert_to_kan`: recursively rewrite standard nn modules to KAN variants

Usage example:

```python
import torch.nn as nn
from converted_KAN.convert import convert_to_kan

model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8 * 16 * 16, 10),
)

kan_model = convert_to_kan(model)  # weights/biases are copied over
```

Notes:
- Conv2d padding_mode must be `"zeros"`.
- MaxPool2d: only dilation=1, ceil_mode=False, return_indices=False.
- AvgPool2d: ceil_mode=False, divisor_override=None (respects count_include_pad).
