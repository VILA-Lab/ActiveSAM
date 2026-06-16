"""Margin-aware background calibration
"""
from __future__ import annotations

import torch


def apply_mabc(
    seg_logits: torch.Tensor,    # [num_queries, H, W]  per-pixel per-class scores
    seg_pred: torch.Tensor,      # [H, W]  argmax indices (overwritten in-place)
    prob_thd: float,             # dataset's background threshold
    bg_idx: int,                 # background class index (typically 0)
    exponent: float = 1.25,      # threshold exponent T = prob_thd ** exponent
) -> torch.Tensor:
    """Apply margin-aware background calibration in place; returns ``seg_pred``."""
    if prob_thd <= 0 or seg_logits.shape[0] < 2:
        return seg_pred

    topk = torch.topk(seg_logits, k=2, dim=0).values   # [2, H, W]
    s1, s2 = topk[0], topk[1]
    margin = (s1 - s2).clamp_min_(0)
    energy = s1 * margin.sqrt()
    threshold = float(prob_thd) ** float(exponent)
    seg_pred[energy < threshold] = bg_idx
    return seg_pred
