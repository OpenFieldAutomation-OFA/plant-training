_base_ = 'dinov2_base.py'

train_dataloader = dict(
    dataset=dict(
        ann_file='annotation/plantclefcaw_train.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation/plantclefcaw_val.txt',
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
