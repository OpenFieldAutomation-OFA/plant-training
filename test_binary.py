_base_ = 'baseline.py'

test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_test.txt',
    ),
)

test_evaluator = dict(_delete_=True, type='BinaryMetric', class_id=2473) # maize
# test_evaluator = dict(_delete_=True, type='BinaryMetric', class_id=2354) # sugar beet