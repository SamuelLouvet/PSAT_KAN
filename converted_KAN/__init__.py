from .linearkan import LinearKAN, Linear
from .conv2dkan import Conv2dKAN
from .avgpool2dkan import AvgPool2dKAN
from .batchnorm2dkan import BatchNorm2d
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
