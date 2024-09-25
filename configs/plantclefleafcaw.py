_base_ = 'dinov2_base.py'
# _base_ = 'dinov2_small.py'

train_dataloader = dict(
    dataset=dict(
        ann_file='annotation/plantclefleafcaw_train.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation/plantclefleafcaw_val.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_test.txt',
    ),
)

test_evaluator = [
  dict(type='Accuracy', topk=(1, 5)),
  dict(type='SingleLabelMetric'),
  dict(type='BinaryMetric', class_id=2354), # sugar beet
  dict(type='BinaryMetric', class_id=2473) # maize
]
