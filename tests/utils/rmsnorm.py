"""RMSNorm module implementation for CS336 Assignment 1.

Implements Root Mean Square Layer Normalization as described in
Zhang et al. (2019), equation 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Given an input vector a ∈ R^{d_model}, RMSNorm rescales each element a_i:
        RMSNorm(a_i) = a_i / RMS(a) * g_i
    where:
        RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
    and g_i is a learnable gain parameter.

    Args:
        d_model: Hidden dimension of the model.
        eps: Epsilon value for numerical stability.
        device: Device to store the parameters on.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Initialize gain parameter to ones (𝟙)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Tensor of same shape as input with RMSNorm applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMS(a) = sqrt(mean(a^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight.to(torch.float32)
        return result.to(in_dtype)
