"""Linear module implementation for CS336 Assignment 1.

Implements a linear transformation without bias, following the interface
of PyTorch's nn.Linear (but without bias).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """A linear transformation module without bias.

    Performs the transformation y = x @ W.T, where W is stored in row-major
    format with shape (out_features, in_features). This is mathematically
    equivalent to y = Wx for column vectors, but uses row-major memory
    ordering consistent with PyTorch conventions.

    Args:
        in_features: Size of the input dimension (d_in).
        out_features: Size of the output dimension (d_out).
        device: Device to store the parameters on.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights: N(0, sigma^2) with sigma^2 = 2 / (d_in + d_out)
        # truncated at [-3*sigma, 3*sigma]
        std = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        return x @ self.weight.T
