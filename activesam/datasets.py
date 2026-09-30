"""Benchmarks not shipped with mmsegmentation"""
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PascalVOC20Dataset(BaseSegDataset):
    """Pascal VOC 2012 without the background class."""
    METAINFO = dict(classes=(
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'))

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=True, **kwargs)


@DATASETS.register_module()
class PascalContext60Dataset(BaseSegDataset):
    """Pascal Context with 59 classes and background."""
    METAINFO = dict(classes=(
        'background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
        'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain',
        'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
        'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
        'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
        'wall', 'water', 'window', 'wood'))

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)


@DATASETS.register_module()
class PascalContext59Dataset(BaseSegDataset):
    """Pascal Context with 59 classes, background ignored."""
    METAINFO = dict(classes=PascalContext60Dataset.METAINFO['classes'][1:])

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=True, **kwargs)


@DATASETS.register_module()
class COCOObjectDataset(BaseSegDataset):
    """COCO-Object: the 80 COCO object classes and background."""
    METAINFO = dict(classes=(
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'))

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)
