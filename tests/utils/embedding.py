"""Embedding module implementation for CS336 Assignment 1.

Implements an embedding lookup layer that maps integer token IDs to
dense vector representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """An embedding lookup module.

    Maps integer token IDs to dense vector representations by indexing
    into an embedding matrix of shape (num_embeddings, embedding_dim).

    Args:
        num_embeddings: Size of the vocabulary (vocab_size).
        embedding_dim: Dimension of the embedding vectors (d_model).
        device: Device to store the parameters on.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize weights: N(0, 1) truncated at [-3, 3]
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: Integer tensor of token IDs with arbitrary shape.

        Returns:
            Tensor of shape (*token_ids.shape, embedding_dim) containing
            the embedding vectors for each token ID.
        """
        return self.weight[token_ids]
