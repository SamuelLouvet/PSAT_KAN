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
    MultiplyKAN,
    ReciprocalKAN,
    MaxKAN,
    ExpKAN,
    SumKAN,
    kan_multiply,
    kan_division,
    kan_reciprocal,
    kan_square,
    kan_scale_quarter,
)
from .relumaxpool2d import PairwiseMaxKAN, MaxReduceLastDimFixedKAN
from .attentionkan import (
    AttentionKAN,
    Attention,
    SelfAttentionKAN,
    SelfAttention,
    MatMulKAN,
    ScaleKAN,
    kan_matmul,
    kan_batched_matmul,
    kan_scale,
)
from .convert import convert_to_kan, to_kan

BatchNorm2d = AffineKAN
