# _base_ = 'baseline.py'
_base_ = 'dinov2_small.py'

# don't freeze backbone and load pretrained classification head
model = dict(
    backbone=dict(
        frozen_stages=0,
    ),
    neck=None,
    head=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/data/saved/20240822_081104_leafcaw_small.pth',
            prefix='head',
        ),
    ),
)

# reduce lr
optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
)
train_cfg = dict(max_epochs=10)

train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='annotation/train_2.txt',
    ),
)
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='annotation/val_2.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/test_2.txt',
    ),
)

default_hooks = dict(
    logger=dict(interval=100)
)

# test_evaluator = [
#   dict(type='Accuracy', topk=(1, 5)),
#   dict(type='SingleLabelMetric'),
# ]