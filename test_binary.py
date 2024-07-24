_base_ = 'baseline.py'

test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_test_small.txt',
    ),
)

test_evaluator = dict(_delete_=True, type='BinaryMetric', class_ids=[2473], increase_output=0.1) # maize
# test_evaluator = dict(_delete_=True, type='BinaryMetric', class_ids=[2354], increase_output=0.1) # sugar beet

# test_evaluator = dict(_delete_=True, type='BinaryMetric', class_ids=[2473], topk=7806, thrs=0.1) # maize
