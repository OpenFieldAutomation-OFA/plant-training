_base_ = 'baseline.py'

train_dataloader = dict(
    dataset=dict(
        ann_file='annotation/train_2.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation/val_2.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_test.txt',
    ),
)