"""Low-resolution preview + quantile pruning of the class vocabulary.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from ..model.processor import BucketPromptProcessor
from ..model.rope_patch import enable_dynamic_vit_rope
from .bucket_scheduler import schedule_sequential


class ActiveClassPruner:
    def __init__(
        self,
        model,
        device,
        num_queries: int,
        query_words: Sequence[str],
        bucket_size: int = 32,
        confidence_threshold: float = 0.3,
        preview_resolution: int = 672,
        quantile_alpha: float = 0.20,
        pruner_max_active: int = 40,
    ):
        self.device = device
        self.num_queries = int(num_queries)
        self.query_words = list(query_words)
        self.bucket_size = int(bucket_size)
        self.quantile_alpha = float(quantile_alpha)
        self.pruner_max_active = int(pruner_max_active)

        # Enable variable-resolution RoPE for the low-resolution preview.
        enable_dynamic_vit_rope()
        self._preview_processor = BucketPromptProcessor(
            model,
            resolution=int(preview_resolution),
            device=device,
            confidence_threshold=confidence_threshold,
        )

    @torch.inference_mode()
    def preview_and_decide(self, image) -> List[int]:
        """Return the sorted list of active class indices for this image."""
        # Skip the preview when the vocabulary already fits under the cap.
        if self.num_queries <= self.pruner_max_active:
            return list(range(self.num_queries))
        return self._decide_active(self._score_presence(image))

    def _score_presence(self, image) -> torch.Tensor:
        state = self._preview_processor.set_image(image)
        scores = torch.zeros(self.num_queries, device=self.device)
        for bucket in schedule_sequential(self.num_queries, self.bucket_size):
            prompts = [self.query_words[i] for i in bucket]
            per_class = self._preview_processor.set_text_prompts_batch(prompts, state)
            for slot, class_idx in enumerate(bucket):
                scores[class_idx] = per_class[slot]["presence_score"].float()
        return scores

    def _decide_active(self, scores: torch.Tensor) -> List[int]:
        tau = float(scores.float().quantile(1.0 - self.quantile_alpha).item())
        idx = torch.where(scores >= tau)[0]
        if idx.numel() == 0:
            # Never produce an empty set; fall back to the top-1 class.
            return [int(scores.argmax().item())]
        if idx.numel() > self.pruner_max_active:
            idx = idx[torch.topk(scores[idx], k=self.pruner_max_active).indices]
        return sorted(int(v.item()) for v in idx)
