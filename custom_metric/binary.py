from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS

@METRICS.register_module()
class BinaryMetric(BaseMetric):
    def __init__(self, class_id: int, thrs: float = 0) -> None:
        super().__init__()
        self.class_id = class_id
        self.thrs = thrs

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_score' in data_sample:
                pred_score = data_sample['pred_score']
                pred_label = pred_score.argmax(dim=0, keepdim=True)
                if pred_score[pred_label] < self.thrs:
                    pred_label = None
            else:
                pred_label = data_sample['pred_label']

            self.results.append({
                'pred_label': pred_label,
                'gt_label': data_sample['gt_label'],
            })

    def compute_metrics(self, results: list) -> dict:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for result in results:
            if result['pred_label'] == self.class_id and result['gt_label'] == self.class_id:
                tp += 1
            if result['pred_label'] != self.class_id and result['gt_label'] != self.class_id:
                tn += 1
            if result['pred_label'] == self.class_id and result['gt_label'] != self.class_id:
                fp += 1
            if result['pred_label'] != self.class_id and result['gt_label'] == self.class_id:
                fn += 1
            
        acc = (tp + tn) / (tp + tn + fp + fn)
        if tp + fp == 0:
            prec = None
        else:
            prec = tp / (tp + fp)
        
        if tp + fn == 0:
            rec = None
        else:
            rec = tp / (tp + fn)
        
        return {'accuracy': acc, 'precision': prec, 'recall': rec,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
