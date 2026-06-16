"""VOC21 (has bg at idx 0)."""
_base_ = './base_config.py'

model = dict(
    classname_path='./configs/cls_voc21.txt',
    prob_thd=0.4,
    bg_idx=0,
)

dataset_type = 'PascalVOCDataset'
data_root = './data/VOC2012'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline,
    ),
)
