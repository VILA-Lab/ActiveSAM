_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_context60.txt', signatures='signatures/context60.npz', background_label=0)

test_dataloader = dict(dataset=dict(
    type='PascalContext60Dataset',
    data_root='data/VOC2010',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
    ann_file='ImageSets/SegmentationContext/val.txt'))
