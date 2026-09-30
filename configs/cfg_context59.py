_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_context59.txt', signatures='signatures/context59.npz')

test_dataloader = dict(dataset=dict(
    type='PascalContext59Dataset',
    data_root='data/VOC2010',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
    ann_file='ImageSets/SegmentationContext/val.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='PackSegInputs')]))
