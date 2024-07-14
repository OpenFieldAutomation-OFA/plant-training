auto_scale_lr = dict(base_batch_size=1024)
data_dir = '/mnt/data'
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=7806,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, type='CheckpointHook'),
    logger=dict(interval=25, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = '/mnt/data/saved/20240709_083652_leafcaw_new.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='base',
        frozen_stages=12,
        img_size=518,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth',
            prefix='backbone',
            type='Pretrained'),
        layer_scale_init_value=1e-05,
        patch_size=14,
        type='VisionTransformer'),
    head=dict(
        in_channels=768,
        loss=dict(type='CrossEntropyLoss'),
        num_classes=7806,
        topk=(
            1,
            5,
        ),
        type='VisionTransformerClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            cls_token=dict(decay_mult=0.0), mask_token=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(T_max=20, by_epoch=True, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=True, seed=0)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1024,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='annotation/caw_test.txt',
        data_prefix='/mnt/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=518,
                type='ResizeEdge'),
            dict(crop_size=518, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=28,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
test_evaluator = dict(
    class_ids=[
        2473,
    ], increase_output=0.05, type='BinaryMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=518,
        type='ResizeEdge'),
    dict(crop_size=518, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
train_dataloader = dict(
    batch_size=1024,
    dataset=dict(
        ann_file='annotation/train.txt',
        data_prefix='/mnt/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=518,
                type='ResizeEdge'),
            dict(crop_size=518, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=28,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=518,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1024,
    dataset=dict(
        ann_file='annotation/val.txt',
        data_prefix='/mnt/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=518,
                type='ResizeEdge'),
            dict(crop_size=518, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=28,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/test_binary'
