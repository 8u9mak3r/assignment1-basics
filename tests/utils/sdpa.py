"""Scaled Dot-Product Attention implementations for CS336 Assignment 1."""

from __future__ import annotations

import math

import torch


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) * V

    Args:
        Q: Query tensor of shape (..., queries, d_k).
        K: Key tensor of shape (..., keys, d_k).
        V: Value tensor of shape (..., keys, d_v).
        mask: Boolean mask of shape (..., queries, keys) where True means
            "mask out" (set to -inf before softmax). Can be None.

    Returns:
        Output tensor of shape (..., queries, d_v).
    """
    d_k = Q.size(-1)
    # Compute attention scores: (..., queries, keys)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ V
