_base_ = 'baseline.py'

test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_test.txt',
    ),
)

test_evaluator = dict(_delete_=True, type='BinaryMetric', class_ids=[2473]) # maize
# test_evaluator = dict(_delete_=True, type='BinaryMetric', class_ids=[2354]) # sugar beet

# test_evaluator = dict(_delete_=True, type='BinaryMetric', class_ids=[2473], topk=7806, thrs=0.25) # maize
