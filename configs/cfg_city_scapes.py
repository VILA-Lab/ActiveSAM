_base_ = './base_config.py'

model = dict(vocabulary='configs/cls_city_scapes.txt', signatures='signatures/city_scapes.npz')

test_dataloader = dict(dataset=dict(
    type='CityscapesDataset',
    data_root='data/CityScapes',
    data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val')))
