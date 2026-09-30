_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_coco_object.txt', signatures='signatures/coco_object.npz',
             background_label=0)

test_dataloader = dict(dataset=dict(
    type='COCOObjectDataset',
    data_root='data/COCOObject',
    data_prefix=dict(img_path='images/val2017', seg_map_path='annotations/val2017'),
    reduce_zero_label=False))
