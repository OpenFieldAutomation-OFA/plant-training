_base_ = 'baseline.py'

train_dataloader = dict(
    num_workers=4,
)
val_dataloader = dict(
    num_workers=4,
)
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        ann_file='annotation/caw_test.txt',
    ),
)

val_evaluator = dict(_delete_=True, type='MaizeMetric')
test_evaluator = dict(_delete_=True, type='MaizeMetric')