"""Multi-Head Self-Attention implementation for CS336 Assignment 1.

Implements causal multi-head self-attention as described in section 3.2.2
of Vaswani et al. (2017), with optional RoPE support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .linear import Linear
from .sdpa import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention module.

    This module performs the attention computation and output projection
    on pre-computed multi-head Q, K, V tensors. The Q/K/V projections
    are handled externally.

    Computes:
        out = W_O * Concat(head_1, ..., head_h)
    where:
        head_i = Attention(Q_i, K_i, V_i)
    with causal masking preventing attending to future tokens.

    Args:
        d_model: Dimensionality of the input and output.
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        rope: Optional RoPE module for rotary position embeddings.
        device: Device to store parameters on.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope

        # Output projection: (d_model, d_model)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            Q: Query tensor of shape (..., num_heads, seq_len, d_k).
            K: Key tensor of shape (..., num_heads, seq_len, d_k).
            V: Value tensor of shape (..., num_heads, seq_len, d_v).
            token_positions: Optional position indices for RoPE, shape (..., seq_len).

        Returns:
            Output tensor of shape (..., seq_len, d_model).
        """
        *batch_dims, num_heads, seq_len, d_k = Q.shape
        _ = num_heads  # should match self.num_heads

        # Apply RoPE to Q and K if provided
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Create causal mask: True for positions that CAN be attended (j <= i)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device),
            diagonal=1,
        )
        mask = ~causal_mask  # True where j <= i (can attend)

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: (..., num_heads, seq_len, d_v) -> (..., seq_len, d_model)
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*batch_dims, seq_len, self.d_model)

        # Output projection
        return self.o_proj(attn_output)
