"""Context memory banks: extra text tokens appended to the prompt sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn


@dataclass
class MemoryBatch:
    """Extra memory tokens for one bucket forward.
    """

    tokens: torch.Tensor   # [M, K, d_text]
    mask: torch.Tensor     # [K, M]  with True = ignore (padding), False = valid


class ContextMemoryBank(nn.Module):

    def precompute_global(
        self,
        query_words: List[str],
        pooled_text: torch.Tensor,   # [num_classes, d_text=1024]
        resizer: nn.Linear,          # SAM 3's 1024 -> 256 projection (reused)
        device: torch.device,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def extra_memory_tokens(
        self,
        bucket_class_slots: List[int],
        device: torch.device,
    ) -> Optional[MemoryBatch]:
        """Memory tokens for the classes in this bucket, or None to inject none."""
        raise NotImplementedError


class CompositeContextBank(ContextMemoryBank):
    """Chain several banks; their tokens are concatenated along the M axis."""

    def __init__(self, banks: List[ContextMemoryBank]):
        super().__init__()
        self.banks = nn.ModuleList(banks)

    def precompute_global(self, query_words, pooled_text, resizer, device, **kwargs):
        for b in self.banks:
            b.precompute_global(query_words, pooled_text, resizer, device, **kwargs)

    def extra_memory_tokens(self, bucket_class_slots, device):
        collected = [b.extra_memory_tokens(bucket_class_slots, device) for b in self.banks]
        collected = [c for c in collected if c is not None]
        if not collected:
            return None
        tokens = torch.cat([c.tokens for c in collected], dim=0)  # cat on M
        mask = torch.cat([c.mask for c in collected], dim=1)      # cat on M
        return MemoryBatch(tokens=tokens, mask=mask)


@torch.inference_mode()
def pooled_text_embeddings(
    model,
    query_words: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Mean-pool each class's pre-transformer token embeddings into one vector.
    """
    out = model.backbone.forward_text(query_words, device=device)
    embeds = out["language_embeds"]          # [seq, K, 1024]
    mask = out["language_mask"]              # [K, seq]  True = pad, False = valid
    valid = (~mask).to(embeds.dtype).transpose(0, 1).unsqueeze(-1)   # [seq, K, 1]
    summed = (embeds * valid).sum(dim=0)                             # [K, 1024]
    counts = valid.sum(dim=0).clamp_min(1.0)                         # [K, 1]
    return summed / counts


def cosine_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalised cosine-similarity matrix for ``x`` of shape ``[K, D]``."""
    x_norm = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x_norm @ x_norm.T


def project_text_to_prompt_dim(x: torch.Tensor, resizer: nn.Linear) -> torch.Tensor:
    """Project pooled 1024-d embeddings to SAM 3's 256-d prompt dimension."""
    return resizer(x)
