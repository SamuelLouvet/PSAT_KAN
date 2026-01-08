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
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

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
        x_hat = (x - mean) / torch.sqrt(var + self.eps) #TODO change the division look in the pdf paper (a+b)^2 + (a-b)^2/4

        gamma = self.weight.view(1, C, 1, 1)
        beta = self.bias.view(1, C, 1, 1)
        y = gamma * x_hat + beta

        return y



