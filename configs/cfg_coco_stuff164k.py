_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_coco_stuff.txt', signatures='signatures/coco_stuff164k.npz')

test_dataloader = dict(dataset=dict(
    type='COCOStuffDataset',
    data_root='data/COCOStuff',
    data_prefix=dict(img_path='images/val2017', seg_map_path='annotations/val2017')))
