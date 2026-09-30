_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_ade20k.txt', signatures='signatures/ade20k.npz')

test_dataloader = dict(dataset=dict(
    type='ADE20KDataset',
    data_root='data/ADE20K',
    data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='PackSegInputs')]))
