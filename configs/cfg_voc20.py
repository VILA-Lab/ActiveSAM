_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_voc20.txt', signatures='signatures/voc20.npz')

test_dataloader = dict(dataset=dict(
    type='PascalVOC20Dataset',
    data_root='data/VOC2012',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
    ann_file='ImageSets/Segmentation/val.txt'))
