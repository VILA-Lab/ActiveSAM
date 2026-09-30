import numpy as np
from mmseg.registry import MODELS
from PIL import Image

from .imagecorruptions import corrupt
from .segmentor import ActiveSAM


@MODELS.register_module()
class ActiveSAMCorrupted(ActiveSAM):
    """ActiveSAM on corrupted inputs: the image is corrupted once, before the preview and the full pass.
    """

    def __init__(self, *args, corruption_type, corruption_severity=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity

    def load_image(self, path):
        image = np.array(super().load_image(path))
        corrupted = corrupt(image, severity=self.corruption_severity, corruption_name=self.corruption_type)
        return Image.fromarray(np.uint8(np.clip(corrupted, 0, 255)))
