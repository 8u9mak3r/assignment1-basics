"""Transformer block and Transformer LM implementations for CS336 Assignment 1."""

from __future__ import annotations

import torch
import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .mha import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .rope import RoPE
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE and SwiGLU.

    Structure:
        y = x + MHA(RMSNorm(x))
        out = y + SwiGLU(RMSNorm(y))

    Args:
        d_model: Dimensionality of the input and output.
        num_heads: Number of attention heads.
        d_ff: Dimensionality of the feed-forward inner layer.
        max_seq_len: Maximum sequence length for RoPE cache.
        theta: RoPE theta parameter.
        device: Device to store parameters on.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        d_k = d_model // num_heads

        # Layer norms
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # Attention sub-modules (container)
        self.attn = nn.Module()
        self.attn.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attn.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attn.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attn.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.attn.rope = RoPE(d_k=d_k, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)

        # FFN sub-modules (container)
        self.ffn = nn.Module()
        self.ffn.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.ffn.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.ffn.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.num_heads = num_heads
        self.d_k = d_k

        # MHA submodule (shares RoPE and output_proj with self.attn)
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=self.attn.rope,
            device=device,
            dtype=dtype,
        )
        self.mha.o_proj = self.attn.output_proj

        # SwiGLU submodule (shares w1/w2/w3 with self.ffn)
        self.swiglu = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.swiglu.w1 = self.ffn.w1
        self.swiglu.w2 = self.ffn.w2
        self.swiglu.w3 = self.ffn.w3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Transformer block.

        Args:
            x: Input tensor of shape (..., seq_len, d_model).

        Returns:
            Output tensor of shape (..., seq_len, d_model).
        """
        *batch_dims, seq_len, _ = x.shape

        # --- First sub-layer: MHA with RoPE ---
        residual = x
        x_norm = self.ln1(x)

        # Project Q, K, V
        Q = self.attn.q_proj(x_norm)
        K = self.attn.k_proj(x_norm)
        V = self.attn.v_proj(x_norm)

        # Reshape to multi-head: (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        # Apply MHA (handles RoPE + causal mask + SDPA + output projection)
        attn_out = self.mha(Q, K, V)

        # Residual connection
        x = residual + attn_out

        # --- Second sub-layer: SwiGLU FFN ---
        residual = x
        x_norm = self.ln2(x)

        ffn_out = self.swiglu(x_norm)

        return residual + ffn_out


class TransformerLM(nn.Module):
    """Full Transformer language model.

    Structure:
        x = Embedding(token_ids)
        for layer in layers:
            x = TransformerBlock(x)
        x = RMSNorm(x)
        logits = LMHead(x)

    Args:
        vocab_size: Size of the vocabulary.
        context_length: Maximum sequence length.
        d_model: Hidden dimension.
        num_layers: Number of Transformer blocks.
        num_heads: Number of attention heads per block.
        d_ff: Feed-forward inner dimension.
        rope_theta: RoPE theta parameter.
        device: Device to store parameters on.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer LM.

        Args:
            token_ids: Integer tensor of shape (batch_size, seq_len).

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
