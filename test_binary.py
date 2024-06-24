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

# test_evaluator = dict(_delete_=True, type='BinaryMetric', class_id=2473) # maize
test_evaluator = dict(_delete_=True, type='BinaryMetric', class_id=2354) # sugar beet