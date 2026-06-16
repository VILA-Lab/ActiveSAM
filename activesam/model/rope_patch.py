"""Variable-resolution RoPE patch for the SAM 3 ViT.
"""

from __future__ import annotations

import math

import torch

from sam3.model import vitdet as _sam3_vitdet

_DYNAMIC_VIT_ROPE_PATCHED = False


def enable_dynamic_vit_rope() -> None:
    """Patch ``sam3.model.vitdet.Attention._apply_rope`` in place; runs only once."""
    global _DYNAMIC_VIT_ROPE_PATCHED
    if _DYNAMIC_VIT_ROPE_PATCHED:
        return

    original_apply_rope = _sam3_vitdet.Attention._apply_rope

    def _dynamic_apply_rope(self, q, k):
        if not self.use_rope:
            return q, k

        # VE / RoPE-real branches: use the original implementation.
        if getattr(self, "use_ve_rope", False) or getattr(self, "use_rope_real", False):
            return original_apply_rope(self, q, k)

        compute_cis = getattr(self, "compute_cis", None)
        if compute_cis is None:
            return original_apply_rope(self, q, k)

        token_count = q.shape[-2] - (1 if getattr(self, "cls_token", False) else 0)
        side = int(math.isqrt(token_count))
        if side * side != token_count:
            # non-square grid: fall back to the original implementation
            return original_apply_rope(self, q, k)

        if self.freqs_cis is None or self.freqs_cis.shape[0] != q.shape[-2]:
            scale_pos = 1.0
            if getattr(self, "rope_interp", False):
                base_size = (
                    self.rope_pt_size[0]
                    if getattr(self, "rope_pt_size", None) is not None
                    else side
                )
                scale_pos = base_size / side

            freqs_cis = compute_cis(end_x=side, end_y=side, scale_pos=scale_pos)
            if getattr(self, "cls_token", False):
                t = torch.zeros(
                    self.head_dim // 2,
                    dtype=torch.float32,
                    device=freqs_cis.device,
                )
                cls_freqs_cis = torch.polar(torch.ones_like(t), t)[None, :]
                freqs_cis = torch.cat([cls_freqs_cis, freqs_cis], dim=0)
            self.freqs_cis = freqs_cis.to(device=q.device)

        return _sam3_vitdet.apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)

    _sam3_vitdet.Attention._apply_rope = _dynamic_apply_rope
    _DYNAMIC_VIT_ROPE_PATCHED = True
