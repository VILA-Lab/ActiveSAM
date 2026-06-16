"""Semantic-neighbour context tokens.

For each class, select the top-M most similar other classes by cosine similarity
over pooled class-name text embeddings, and form similarity-weighted memory
tokens from their pooled embeddings projected into SAM 3's 256-d prompt
dimension.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .memory_bank import (
    ContextMemoryBank,
    MemoryBatch,
    cosine_matrix,
    project_text_to_prompt_dim,
)


class SemanticNeighborBank(ContextMemoryBank):
    """Top-M nearest classes by cosine similarity in pooled-text-embedding space."""

    def __init__(self, num_tokens: int = 2, cosine_temperature: float = 1.0,
                 skip_self: bool = True):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.temperature = float(cosine_temperature)
        self.skip_self = bool(skip_self)
        # Populated in precompute_global: [num_classes, M, 256].
        self.register_buffer("_tokens", torch.empty(0), persistent=False)

    @torch.inference_mode()
    def precompute_global(self, query_words, pooled_text, resizer, device, **kwargs):
        """Build the [num_classes, M, 256] memory-token bank."""
        M = self.num_tokens

        sim = cosine_matrix(pooled_text)         # [K, K]
        if self.skip_self:
            sim.fill_diagonal_(-float("inf"))

        topk_vals, topk_idx = sim.topk(M, dim=1)   # [K, M]
        weights = torch.softmax(
            topk_vals / max(self.temperature, 1e-6), dim=1
        )                                          # [K, M]
        tokens_1024 = pooled_text[topk_idx] * weights.unsqueeze(-1)  # [K, M, d_text]

        tokens_256 = project_text_to_prompt_dim(tokens_1024, resizer)  # [K, M, 256]
        self._tokens = tokens_256.to(device=device, dtype=torch.float32).contiguous()

    def extra_memory_tokens(self, bucket_class_slots: List[int],
                            device: torch.device) -> Optional[MemoryBatch]:
        if self._tokens.numel() == 0:
            return None
        K = len(bucket_class_slots)
        M = self.num_tokens
        d = self._tokens.size(-1)

        # [M, K, d] in seq-first layout; every neighbour token is valid.
        tokens = torch.zeros(M, K, d, device=device, dtype=self._tokens.dtype)
        for k, cls_idx in enumerate(bucket_class_slots):
            tokens[:, k, :] = self._tokens[cls_idx]
        mask = torch.zeros(K, M, device=device, dtype=torch.bool)   # False = attend
        return MemoryBatch(tokens=tokens, mask=mask)
