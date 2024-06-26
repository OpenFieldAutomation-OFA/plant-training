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
        ann_file='annotation_binary/train_2.txt',
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file='annotation_binary/val_2.txt',
    ),
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotation_binary/test_2.txt',
    ),
)

val_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  dict(type='SingleLabelMetric', items=['precision', 'recall']),
]
test_evaluator = [
  dict(type='Accuracy', topk=(1, )),
  dict(type='SingleLabelMetric', items=['precision', 'recall']),
]
