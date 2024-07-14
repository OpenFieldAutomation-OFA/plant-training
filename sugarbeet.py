_base_ = 'baseline.py'

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
        ann_file='annotation_sugarbeet/train_2.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation_sugarbeet/val_2.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation_sugarbeet/caw_test.txt',
    ),
)

val_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  dict(type='SingleLabelMetric', average=None),
]
test_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  dict(type='SingleLabelMetric', average=None),
  dict(type='BinaryMetric', class_ids=[1], topk=2, thrs=0.3)
]
