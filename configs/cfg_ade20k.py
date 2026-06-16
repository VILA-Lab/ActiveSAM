"""ADE20K."""
_base_ = './base_config.py'

model = dict(
    classname_path='./configs/cls_ade20k.txt',
)

dataset_type = 'ADE20KDataset'
data_root = './data/ADE20K'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]

test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline,
    ),
)
