from __future__ import annotations

from typing import Dict, List, Optional

import torch

from sam3.model.data_misc import FindStage, interpolate
from sam3.model.sam3_image_processor import Sam3Processor

from ..prompt_expansion.memory_bank import ContextMemoryBank, MemoryBatch


class BucketPromptProcessor(Sam3Processor):
    @torch.inference_mode()
    def set_text_prompts_batch(
        self,
        prompts: List[str],
        state: Dict,
        bucket_class_slots: Optional[List[int]] = None,
        memory_bank: Optional[ContextMemoryBank] = None,
    ) -> List[Dict]:
        assert prompts, "prompts must be a non-empty list of class names"
        if "backbone_out" not in state:
            raise ValueError("call set_image before set_text_prompts_batch")

        K = len(prompts)

        # Encode K text prompts in one forward (the text encoder is batch-aware).
        text_outputs = self.model.backbone.forward_text(prompts, device=self.device)
        state["backbone_out"].update(text_outputs)

        # Optionally append the context memory tokens to the text sequence.
        if memory_bank is not None and bucket_class_slots is not None:
            mb: Optional[MemoryBatch] = memory_bank.extra_memory_tokens(
                bucket_class_slots, self.device
            )
            if mb is not None:
                lang = state["backbone_out"]["language_features"]   # [seq, K, 256]
                mask = state["backbone_out"]["language_mask"]       # [K, seq]
                extra_tokens = mb.tokens.to(dtype=lang.dtype, device=lang.device)  # [M, K, 256]
                extra_mask = mb.mask.to(device=mask.device)         # [K, M]
                state["backbone_out"]["language_features"] = torch.cat(
                    [lang, extra_tokens], dim=0)
                state["backbone_out"]["language_mask"] = torch.cat(
                    [mask, extra_mask], dim=1)

        # Replicate the geometric prompt to the text batch size.
        state["geometric_prompt"] = self.model._get_dummy_prompt(num_prompts=K)

        # bs=K FindStage pointing all K text prompts at image 0 (the cached one).
        find_stage_batch = FindStage(
            img_ids=torch.tensor([0] * K, device=self.device, dtype=torch.long),
            text_ids=torch.tensor(list(range(K)), device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage_batch,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        img_h = state["original_height"]
        img_w = state["original_width"]

        pred_only_probs = outputs["pred_logits"].sigmoid().squeeze(-1)              # [K, 200]
        presence_score_all = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)   # [K, 1, 1]
        all_probs = (pred_only_probs.unsqueeze(-1) * presence_score_all).squeeze(-1)  # [K, 200]

        sem_upsampled = interpolate(
            outputs["semantic_seg"], (img_h, img_w), mode="bilinear", align_corners=False
        ).sigmoid()                                                                # [K, 1, H, W]

        per_class_states: List[Dict] = []
        for k in range(K):
            out_masks = outputs["pred_masks"][k]           # [200, H_out, W_out]
            out_probs_k = all_probs[k]                     # [200]  pred * presence
            presence_k = presence_score_all[k].squeeze()   # scalar

            keep = out_probs_k > self.confidence_threshold
            out_probs_k = out_probs_k[keep]
            out_masks = out_masks[keep]

            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()                                    # [n_keep, 1, H, W]

            per_class_states.append({
                "masks_logits": out_masks,
                "semantic_mask_logits": sem_upsampled[k],
                "presence_score": presence_k,
                "object_score": out_probs_k,
            })

        return per_class_states
