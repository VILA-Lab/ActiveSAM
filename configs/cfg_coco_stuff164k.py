"""COCO-Stuff-164K (170 classes; no bg)."""
_base_ = './base_config.py'

model = dict(
    classname_path='./configs/cls_coco_stuff.txt',
)

dataset_type = 'COCOStuffDataset'
data_root = './data/COCOStuff'

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
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline,
    ),
)
