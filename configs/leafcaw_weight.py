_base_ = 'baseline.py'

class_weights = [1] * 7806
class_weights[2354] = 2 # sugar beet
class_weights[2473] = 2 # maize

model = dict(
    head=dict(
        loss=dict(type='CrossEntropyLoss', class_weight=class_weights),
    ),
)

train_dataloader = dict(
    dataset=dict(
        ann_file='annotation/train_2.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation/caw_val.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation/test_2.txt',
    ),
)

val_evaluator = [
  dict(type='BinaryMetric', class_ids=[2473]), # maize
  dict(type='BinaryMetric', class_ids=[2354]), # maize
  dict(type='SingleLabelMetric'),
]
test_evaluator = [
  dict(type='Accuracy', topk=(1, 5)),
  dict(type='SingleLabelMetric'),
]