"""Softmax implementations for CS336 Assignment 1."""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute softmax along the specified dimension with numerical stability.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of same shape as input with softmax applied along `dim`.
    """
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
