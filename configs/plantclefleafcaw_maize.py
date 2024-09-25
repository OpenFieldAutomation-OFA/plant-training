_base_ = 'dinov2_base.py'

model = dict(
    head=dict(
        num_classes=2,
        topk=(1, ),
    ),
)

data_preprocessor = dict(
    num_classes=2,
)

train_dataloader = dict(
    dataset=dict(
        ann_file='annotation/plantclefleafcaw_maize_train.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation/plantclefleafcaw_maize_val.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_maize_test.txt',
    ),
)

val_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  dict(type='SingleLabelMetric', average=None),
]
test_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  dict(type='SingleLabelMetric', average=None),
  dict(type='BinaryMetric', class_id=1)
]
