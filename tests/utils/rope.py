"""RoPE (Rotary Position Embedding) implementation for CS336 Assignment 1.

Implements Rotary Position Embeddings as described in Su et al. (2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies pairwise rotation matrices to query/key vectors to inject
    positional information. For a token at position i, pairs of embedding
    dimensions (2k-1, 2k) are rotated as 2D vectors by angle:
        θ_{i,k} = i / Θ^((2k-2)/d)

    Args:
        d_k: Embedding dimension of the query/key vectors.
        theta: The Θ parameter controlling the base rotation frequency.
        max_seq_len: Maximum sequence length to pre-compute cos/sin caches for.
        device: Device to store the cache on.
        dtype: Data type of the cache.
    """

    def __init__(
        self,
        d_k: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Precompute the frequency for each pair of dimensions
        # freq_k = 1 / theta^(2k / d_k) for k = 0, 1, ..., d_k/2 - 1
        k = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        freq = 1.0 / (theta ** (2 * k / d_k))

        # Precompute cos and sin for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # angles: (max_seq_len, d_k/2)
        angles = positions[:, None] * freq[None, :]

        self.register_buffer("cos_cached", torch.cos(angles).to(dtype))
        self.register_buffer("sin_cached", torch.sin(angles).to(dtype))

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply RoPE to the input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k).
            token_positions: Optional tensor of shape (..., seq_len) with position
                indices. If None, uses sequential positions 0, 1, ..., seq_len-1.

        Returns:
            Tensor of same shape as input with RoPE applied.
        """
        *batch_dims, seq_len, d_k = x.shape

        if token_positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            positions = token_positions

        # Gather cos/sin for the requested positions
        # positions shape may be (batch, seq_len) or (seq_len,)
        cos = self.cos_cached[positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[positions]  # (..., seq_len, d_k/2)

        # Reshape x into pairs: (..., seq_len, d_k/2, 2)
        x_pairs = x.reshape(*batch_dims, seq_len, d_k // 2, 2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]

        # Apply rotation:
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        result = torch.stack([x1_rotated, x2_rotated], dim=-1)
        return result.reshape(*batch_dims, seq_len, d_k)
