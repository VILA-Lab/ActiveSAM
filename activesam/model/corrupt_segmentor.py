"""ActiveSAM under ImageNet-C input corruption.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from mmseg.registry import MODELS

from .segmentor import ActiveSAM


def _build_corrupt_fn(corruption_type: str, severity: int):
    """Return a PIL->PIL callable applying the ImageNet-C corruption (on RGB).
    """
    from ..imagecorruptions import corrupt as _ic_corrupt

    sev = max(1, min(5, int(severity)))

    def _fn(pil_img):
        arr = np.array(pil_img.convert("RGB"))
        out = _ic_corrupt(arr, severity=sev, corruption_name=corruption_type)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    return _fn


@MODELS.register_module()
class ActiveSAMCorrupted(ActiveSAM):
    """ActiveSAM with an input-side ImageNet-C corruption hook.

    With ``corruption_type=None`` it behaves exactly like ``ActiveSAM``.
    """

    def __init__(self, *args, corruption_type: Optional[str] = None,
                 corruption_severity: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption_type = corruption_type
        self.corruption_severity = int(corruption_severity)
        self._corrupt_fn = (
            _build_corrupt_fn(corruption_type, self.corruption_severity)
            if corruption_type else None
        )

    def _inference_single_view(self, image):
        if self._corrupt_fn is not None:
            image = self._corrupt_fn(image)
        return super()._inference_single_view(image)
