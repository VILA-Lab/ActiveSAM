"""ActiveSAM open-vocabulary segmentor on a frozen SAM 3.

Combines the four parts of the method: contextual prompt expansion (CPE),
preview-driven class selection, bucketed full-resolution decoding, and
margin-aware background calibration (MABC).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from mmseg.registry import MODELS

from .base import FrozenSAM3Segmentor
from .processor import BucketPromptProcessor
from .text_cache import enable_text_cache
from .skip_seg_head import install_skip_patch, skip_seg_head_ctx
from ..prompt_expansion.cpe import (
    install_lexical_canonicalizer,
    build_context_memory_bank,
)
from ..prompt_expansion.memory_bank import pooled_text_embeddings
from ..pruning.pruner import ActiveClassPruner
from ..pruning.bucket_scheduler import schedule_sequential
from ..background.mabc import apply_mabc


@MODELS.register_module()
class ActiveSAM(FrozenSAM3Segmentor):
    def __init__(
        self,
        *args,
        neighbor_tokens: int = 2,
        hypernym_tokens: int = 2,
        bucket_size: int = 32,
        preview_resolution: int = 672,
        quantile_alpha: float = 0.20,
        pruner_max_active: int = 40,
        mabc_exponent: float = 1.25,
        **kwargs,
    ):
        # Enable the text cache before any text is encoded.
        enable_text_cache()
        super().__init__(*args, **kwargs)

        underlying_model = self.processor.model
        # Swap the per-class processor for the bucket-batched one.
        self.processor = BucketPromptProcessor(
            underlying_model,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
        self.bucket_size = int(bucket_size)
        self.mabc_exponent = float(mabc_exponent)

        # Contextual Prompt Expansion: lexical canonicalization + context bank.
        install_lexical_canonicalizer(underlying_model, self, universal=True)
        self.memory_bank = build_context_memory_bank(neighbor_tokens, hypernym_tokens)
        self._precompute_memory_bank(underlying_model)

        # Preview-driven class pruner.
        self.pruner = ActiveClassPruner(
            model=underlying_model,
            device=self.device,
            num_queries=self.num_queries,
            query_words=self.query_words,
            bucket_size=self.bucket_size,
            confidence_threshold=self.confidence_threshold,
            preview_resolution=preview_resolution,
            quantile_alpha=quantile_alpha,
            pruner_max_active=pruner_max_active,
        )

        # Skip the segmentation head during the preview pass.
        self.enable_skip_seg_head = int(self.num_queries) > int(self.bucket_size)
        install_skip_patch(underlying_model)
        if self.enable_skip_seg_head:
            base_preview = self.pruner.preview_and_decide

            def preview_with_skip(image):
                with skip_seg_head_ctx(underlying_model):
                    return base_preview(image)

            self.pruner.preview_and_decide = preview_with_skip

    @torch.inference_mode()
    def _precompute_memory_bank(self, model):
        """Pool each class's text embedding once and hand it to the context bank."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pooled = pooled_text_embeddings(model, self.query_words, self.device)
        pooled = pooled.to(dtype=torch.float32)
        resizer = model.backbone.language_backbone.resizer   # nn.Linear(1024, 256)
        self.memory_bank.precompute_global(
            self.query_words, pooled, resizer, self.device, model=model,
        )

    def _assign_background(self, seg_logits, seg_pred):
        """Margin-aware background calibration."""
        return apply_mabc(
            seg_logits, seg_pred,
            prob_thd=self.prob_thd, bg_idx=self.bg_idx, exponent=self.mabc_exponent,
        )

    def _inference_single_view(self, image):
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)
        presence = torch.zeros(self.num_queries, device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Cheap low-resolution preview prunes the vocabulary to the active set.
            active = self.pruner.preview_and_decide(image)
            state = self.processor.set_image(image)

            # Decode the active classes at full resolution, one bucket per forward.
            for bucket in schedule_sequential(len(active), self.bucket_size):
                classes = [active[s] for s in bucket]
                prompts = [self.query_words[c] for c in classes]
                per_class = self.processor.set_text_prompts_batch(
                    prompts, state,
                    bucket_class_slots=classes,
                    memory_bank=self.memory_bank,
                )
                for slot, c in enumerate(classes):
                    inf = per_class[slot]
                    self._fuse_instance_and_semantic(seg_logits, c, inf, h, w)
                    presence[c] = inf["presence_score"]

        # Presence gating.
        seg_logits = seg_logits * presence[:, None, None]
        return seg_logits

    def _fuse_instance_and_semantic(self, seg_logits, query_idx, inf, h, w):
        """Fuse the instance and semantic heads by pixelwise maximum.

        Presence gating is applied once, later, in ``_inference_single_view``.
        """
        # Instance head: max over the surviving instance masks (scored by object
        # confidence).
        masks_logits = inf["masks_logits"]
        if masks_logits.shape[0] > 0:
            object_scores = inf["object_score"]
            for inst_id in range(masks_logits.shape[0]):
                instance_logits = masks_logits[inst_id].squeeze()
                instance_score = object_scores[inst_id]
                if instance_logits.shape != (h, w):
                    instance_logits = F.interpolate(
                        instance_logits.view(1, 1, *instance_logits.shape),
                        size=(h, w), mode="bilinear", align_corners=False,
                    ).squeeze()
                inst_contrib = instance_logits * instance_score
                seg_logits[query_idx] = torch.max(seg_logits[query_idx], inst_contrib)

        # Semantic head.
        sem = inf["semantic_mask_logits"]
        if sem.shape[-2:] != (h, w):
            if sem.dim() == 2:
                sem = sem.unsqueeze(0).unsqueeze(0)
            elif sem.dim() == 3:
                sem = sem.unsqueeze(0)
            sem = F.interpolate(
                sem, size=(h, w), mode="bilinear", align_corners=False).squeeze()
        seg_logits[query_idx] = torch.max(seg_logits[query_idx], sem)
