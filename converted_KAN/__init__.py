from .linearkan import LinearKAN, Linear
from .conv2dkan import Conv2dKAN
from .avgpool2dkan import AvgPool2dKAN
from .batchnorm2dkan import AffineKAN
from .relumaxpool2d import ReLUMaxPool2dKAN, ReLUMaxPool2d
from .softmaxkan import (
    SoftmaxKAN,
    Softmax,
    DivisionKAN,
    Division,
    kan_multiply,
    kan_division,
    kan_reciprocal,
    kan_square,
    kan_scale_quarter,
)
from .attentionkan import (
    AttentionKAN,
    Attention,
    SelfAttentionKAN,
    SelfAttention,
    kan_matmul,
    kan_batched_matmul,
    kan_scale,
)
from .convert import convert_to_kan, to_kan
from .quantization import quantize_model, quantize_tensor
from .accuracy_cost import evaluate_accuracy, accuracy_cost_sweep
from .int_ops import quantize_to_int, dequantize_from_int, kan_multiply_int
from .int_layers import LinearKANInt, Conv2dKANInt, ReLUInt
from .int_convert import convert_to_int_kan, IntKANWrapper

BatchNorm2d = AffineKAN
