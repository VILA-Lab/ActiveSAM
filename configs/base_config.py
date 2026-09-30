model = dict(
    type='ActiveSAM',
    sam3_checkpoint='weights/sam3/sam3.pt',
    resolution=1008,
    # evidence-proportional grounding
    preview_resolution=336,
    preview_layers=2,
    preview_queries=50,
    tau=0.1,
    bucket_size=32,
    confidence_threshold=0.3,
    # contextual prompt expansion
    neighbor_tokens=2,
    hypernym_tokens=2,
    canonical_names=True,
    # exclusive concept decoding
    sigma=0.15,
    background_offset=1.0,
)

# Label-free calibration (calibrate.py): only need to use small subset, around 10k, even 2k still ok. In the official design, around 10k is used.
calibration = dict(
    pool=dict(
        type='COCOStuffDataset',
        data_root='data/COCOStuff',
        data_prefix=dict(img_path='images/train2017', seg_map_path='annotations/train2017'),
        pipeline=[dict(type='LoadImageFromFile'), dict(type='PackSegInputs')]),
    stride=12,
    core_high=0.5,
    core_low=0.3,
    min_pixels=2000,
)

test_pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'), dict(type='PackSegInputs')]
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(pipeline=test_pipeline))
test_evaluator = dict(type='ClassIoUMetric', iou_metrics=['mIoU'])
test_cfg = dict(type='TestLoop')

default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
log_processor = dict(by_epoch=False)
log_level = 'INFO'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))
