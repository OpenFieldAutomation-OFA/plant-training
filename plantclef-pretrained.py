_base_ = 'mmpretrain/configs/_base_/default_runtime.py'

default_hooks = dict(
    logger=dict(interval=500),
    visualization=dict(type='VisualizationHook', enable=True)
)
randomness = dict(deterministic=True, seed=0)

model = dict(
    type='TimmClassifier',
    model_name='vit_base_patch14_reg4_dinov2.lvd142m',
    num_classes=7806,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/mnt/data/plantclef/pretrained/model_best.pth.tar'
    )
)

data_preprocessor = dict(
    num_classes=7806,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        backend='pillow',
        interpolation='bicubic',
        scale=518
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4
    ),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge',
         backend='pillow',
         edge='short',
         interpolation='bicubic',
         scale=518),
    dict(type='CenterCrop', crop_size=518),
    dict(type='PackInputs'),
]

data_dir = '/mnt/data/caw/classification'

train_dataloader = dict(
    batch_size=64,
    num_workers=10,
    dataset=dict(
        type='CustomDataset',
        data_prefix=data_dir,
        ann_file='annotation/caw_train.txt',
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)
val_dataloader = dict(
    batch_size=128,
    num_workers=10,
    dataset=dict(
        type='CustomDataset',
        data_prefix=data_dir,
        ann_file='annotation/caw_val.txt',
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)
test_dataloader = dict(
    batch_size=128,
    num_workers=10,
    dataset=dict(
        type='CustomDataset',
        data_prefix=data_dir,
        ann_file='annotation/caw_test.txt',
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = dict(type='Accuracy', topk=(1, 5))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=1e-3,  # base_batch_size=1024
                   type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }
    )
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=10,
    by_epoch=True,
    begin=0,
    end=10
)

train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Will automatically scale learning rate to batch_size
auto_scale_lr = dict(base_batch_size=1024)
