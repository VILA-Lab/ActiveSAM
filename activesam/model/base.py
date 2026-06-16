"""Base open-vocabulary segmentor on a frozen SAM 3.
"""
from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmengine.structures import PixelData
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_SAM3_CKPT = os.path.join(_PKG_ROOT, "weights", "sam3", "sam3.pt")
_DEFAULT_BPE = os.path.join(_PKG_ROOT, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")


class FrozenSAM3Segmentor(BaseSegmentor):
    def __init__(
        self,
        classname_path,
        device=torch.device("cuda"),
        prob_thd=0.0,
        bg_idx=0,
        confidence_threshold=0.5,
        sam3_ckpt_path=None,
        **kwargs,
    ):
        super().__init__()

        self.device = device

        model = build_sam3_image_model(
            bpe_path=_DEFAULT_BPE,
            checkpoint_path=sam3_ckpt_path or _DEFAULT_SAM3_CKPT,
            device="cuda",
        )

        self.processor = Sam3Processor(
            model, confidence_threshold=confidence_threshold, device=device,
        )
        self.query_words, self.query_idx = load_class_vocabulary(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.confidence_threshold = confidence_threshold

    def _inference_single_view(self, image):
        raise NotImplementedError   # supplied by ActiveSAM

    def _assign_background(self, seg_logits, seg_pred):
        """Assign the background label after the argmax (supplied by ActiveSAM)."""
        raise NotImplementedError

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            image = Image.open(meta.get("img_path")).convert("RGB")
            ori_shape = meta["ori_shape"]

            seg_logits = self._inference_single_view(image)

            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), size=ori_shape,
                    mode="bilinear", align_corners=False,
                ).squeeze(0)

            # Collapse per-alias query logits onto dataset class indices.
            if self.num_cls != self.num_queries:
                sl = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (sl * cls_index).max(1)[0]

            seg_pred = torch.argmax(seg_logits, dim=0)
            seg_pred = self._assign_background(seg_logits, seg_pred)

            data_samples[i].set_data({
                "seg_logits": PixelData(**{"data": seg_logits}),
                "pred_sem_seg": PixelData(**{"data": seg_pred.unsqueeze(0)}),
            })

        return data_samples

    def _forward(self, data_samples): ...
    def inference(self, img, batch_img_metas): ...
    def encode_decode(self, inputs, batch_img_metas): ...
    def extract_feat(self, inputs): ...
    def loss(self, inputs, data_samples): ...


def load_class_vocabulary(path):
    """Read a class-name file into (query_words, query_idx).

    Each line is one class; comma-separated aliases on a line share that class's
    index, so ``query_idx`` maps each alias-level query back to its class.
    """
    with open(path, "r") as f:
        name_sets = f.readlines()
    class_names, class_indices = [], []
    for idx in range(len(name_sets)):
        names_i = [n.strip() for n in name_sets[idx].split(",")]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace("\n", "") for item in class_names]
    return class_names, class_indices
