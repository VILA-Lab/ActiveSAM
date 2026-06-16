"""Per-prompt text-embedding cache for SAM 3.
"""

from __future__ import annotations

import torch

from sam3.model.vl_combiner import SAM3VLBackbone


_TEXT_CACHE_PATCHED = False


def enable_text_cache() -> None:
    """Install per-prompt caching on ``SAM3VLBackbone.forward_text``."""
    global _TEXT_CACHE_PATCHED
    if _TEXT_CACHE_PATCHED:
        return

    original = SAM3VLBackbone.forward_text

    def cached_forward_text(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device="cuda",
    ):
        if input_boxes is not None or additional_text is not None:
            return original(
                self,
                captions,
                input_boxes=input_boxes,
                additional_text=additional_text,
                device=device,
            )

        cache = getattr(self, "_prompt_text_cache", None)
        if cache is None:
            cache = {}
            self._prompt_text_cache = cache

        captions_list = list(captions)
        missing_idx = [i for i, p in enumerate(captions_list) if p not in cache]

        if missing_idx:
            missing_prompts = [captions_list[i] for i in missing_idx]
            fresh = original(
                self,
                missing_prompts,
                input_boxes=None,
                additional_text=None,
                device=device,
            )
            # Store each prompt's slice, detached and cloned.
            for local_i, global_i in enumerate(missing_idx):
                p = captions_list[global_i]
                cache[p] = {
                    "language_features": fresh["language_features"][:, local_i:local_i + 1].detach().clone(),
                    "language_mask": fresh["language_mask"][local_i:local_i + 1].detach().clone(),
                    "language_embeds": fresh["language_embeds"][:, local_i:local_i + 1].detach().clone(),
                }

        # Reassemble the [seq, K, ...] batched tensors in prompt order.
        return {
            "language_features": torch.cat(
                [cache[p]["language_features"] for p in captions_list], dim=1
            ),
            "language_mask": torch.cat(
                [cache[p]["language_mask"] for p in captions_list], dim=0
            ),
            "language_embeds": torch.cat(
                [cache[p]["language_embeds"] for p in captions_list], dim=1
            ),
        }

    SAM3VLBackbone.forward_text = cached_forward_text
    _TEXT_CACHE_PATCHED = True
