_base_ = 'baseline.py'

train_dataloader = dict(
    dataset=dict(
        ann_file='annotation/leaf_train.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation/leaf_val.txt',
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
]