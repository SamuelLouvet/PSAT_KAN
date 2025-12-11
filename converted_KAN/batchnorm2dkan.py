# file: lowlevel_batchnorm2d.py

from typing import Optional
import torch
from torch import nn


class BatchNorm2d(nn.Module):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            # gamma et beta
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        
        self.register_buffer("running_mean", None)
        self.register_buffer("running_var", None)
        self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W)
        """
        assert x.dim() == 4, "..."
        N, C, H, W = x.shape
        assert C == self.num_features, "..."

        mean = self.running_mean
        var = self.running_var

        mean = mean.view(1, C, 1, 1)
        var = var.view(1, C, 1, 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        gamma = self.weight.view(1, C, 1, 1)
        beta = self.bias.view(1, C, 1, 1)
        y = gamma * x_hat + beta

        return y



