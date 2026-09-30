_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_voc21.txt', signatures='signatures/voc21.npz', background_label=0)

test_dataloader = dict(dataset=dict(
    type='PascalVOCDataset',
    data_root='data/VOC2012',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
    ann_file='ImageSets/Segmentation/val.txt'))
