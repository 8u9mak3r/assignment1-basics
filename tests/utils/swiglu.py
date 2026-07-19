"""SwiGLU and SiLU implementations for CS336 Assignment 1.

Implements the SiLU activation function and the SwiGLU feed-forward network
as used in modern LLMs like Llama 3 and Qwen 2.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU (Swish) activation function element-wise.

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of same shape with SiLU applied.
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Computes:
        FFN(x) = W2(SiLU(W1*x) ⊙ W3*x)

    where W1, W3 ∈ R^{d_ff × d_model} and W2 ∈ R^{d_model × d_ff}.

    Args:
        d_model: Dimensionality of the input and output.
        d_ff: Dimensionality of the inner feed-forward layer.
        device: Device to store the parameters on.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # W1, W3: (d_ff, d_model), W2: (d_model, d_ff) — use Linear (no bias)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).
        """
        # FFN(x) = W2(SiLU(W1*x) ⊙ W3*x)
        gate = silu(self.w1(x))
        linear = self.w3(x)
        return self.w2(gate * linear)
