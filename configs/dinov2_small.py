_base_ = '../mmpretrain/configs/_base_/default_runtime.py'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='dinov2-small',
        img_size=518,
        patch_size=14,
        layer_scale_init_value=1e-5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth',
            prefix='backbone',
        ),
        frozen_stages=12,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=7806,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss'),
        topk=(1, 5),
    ),
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
    batch_size=1024,
    num_workers=20,
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
    batch_size=1024,
    num_workers=20,
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
    batch_size=1024,
    num_workers=20,
    dataset=dict(
        type='CustomDataset',
        data_prefix=data_dir,
        ann_file='annotation/test.txt',
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_evaluator = [
  dict(type='Accuracy', topk=(1, 5)),
  dict(type='SingleLabelMetric'),
]
test_evaluator = [
  dict(type='Accuracy', topk=(1, 5)),
  dict(type='SingleLabelMetric'),
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=1e-4,  # auto_scale_lr below
                   type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }
    )
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=20,
    )
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()
default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=5),
    logger=dict(interval=25)
)
randomness = dict(deterministic=True, seed=0)

# Will automatically scale learning rate to batch_size
auto_scale_lr = dict(base_batch_size=1024)
