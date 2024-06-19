_base_ = 'mmpretrain/configs/_base_/default_runtime.py'

# model settings
model = dict(
    type='TimmClassifier',
    model_name='vit_base_patch14_reg4_dinov2.lvd142m',
    num_classes=7806,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/mnt/data/pretrained/dinov2.pth'
    )
)

# dataset settings
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
data_dir = '/mnt/data'
train_dataloader = dict(
    batch_size=16,
    num_workers=48,
    dataset=dict(
        type='CustomDataset',
        data_prefix=data_dir,
        ann_file='annotation/train.txt',
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
        ann_file='annotation/val.txt',
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
        ann_file='annotation/plantclef_val.txt',
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = dict(type='Accuracy', topk=(1, 5))

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=1e-3,  # auto_scale_lr below
                   type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }
    )
)

# learning rate scheduler
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
)

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)
val_cfg = dict()
test_cfg = dict()
default_hooks = dict(checkpoint=dict(max_keep_ckpts=5))
randomness = dict(deterministic=True, seed=0)

# Will automatically scale learning rate to batch_size
auto_scale_lr = dict(base_batch_size=1024)
