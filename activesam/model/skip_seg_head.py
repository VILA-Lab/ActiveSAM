"""Skip the segmentation head during the pruner's preview pass.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


_PATCH_MARKER = "_cp_skip_seg_head_patched"


def install_skip_patch(model: Any) -> None:
    """Install the segmentation-head skip patch on the SAM 3 image model."""
    cls = type(model)
    if getattr(cls, _PATCH_MARKER, False):
        return
    orig = cls._run_segmentation_heads

    def patched_run_segmentation_heads(self, out, backbone_out, img_ids,
                                       vis_feat_sizes, encoder_hidden_states,
                                       prompt, prompt_mask, hs, **kw):
        if getattr(self, "_skip_seg_head_preview", False):
            # Emit dummy 1x1 masks; the preview only uses presence_score.
            device = hs.device
            dtype = hs.dtype
            bs = hs.size(1)
            nq = hs.size(2)
            out["pred_masks"] = torch.zeros(bs, nq, 1, 1, device=device, dtype=dtype)
            out["semantic_seg"] = torch.zeros(bs, 1, 1, 1, device=device, dtype=dtype)
            return
        return orig(self, out, backbone_out, img_ids, vis_feat_sizes,
                    encoder_hidden_states, prompt, prompt_mask, hs, **kw)

    cls._run_segmentation_heads = patched_run_segmentation_heads
    setattr(cls, _PATCH_MARKER, True)


@contextmanager
def skip_seg_head_ctx(model: Any):
    """Toggle the preview-skip flag on ``model`` (an instance) for the block."""
    prev = getattr(model, "_skip_seg_head_preview", False)
    model._skip_seg_head_preview = True
    try:
        yield
    finally:
        model._skip_seg_head_preview = prev
