"""COCO-Object (has bg at idx 0). prob_thd=0.4."""
_base_ = './base_config.py'

model = dict(
    classname_path='./configs/cls_coco_object.txt',
    prob_thd=0.4,
    bg_idx=0,
)

dataset_type = 'COCOObjectDataset'
data_root = './data/COCOObject'

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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline,
    ),
)
