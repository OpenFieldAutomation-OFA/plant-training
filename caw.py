_base_ = 'baseline.py'

train_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_train.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_val.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_test.txt',
    ),
)