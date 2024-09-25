from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from collections import Counter

from mmpretrain.registry import METRICS

@METRICS.register_module()
class BinaryMetric(BaseMetric):
    def __init__(self, class_id: int, increase_output=None) -> None:
        super().__init__()
        self.class_id = class_id
        self.increase_output = increase_output

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_score' in data_sample:
                pred_score = data_sample['pred_score'].cpu()

            self.results.append({
                'pred_score': pred_score,
                'gt_label': data_sample['gt_label'],
            })

    def compute_metrics(self, results: list) -> dict:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        fns = []
        for result in results:
            pred_score = result['pred_score']
            if self.increase_output is not None:
                pred_score[self.class_id] += self.increase_output

            top_class = pred_score.argmax().item()

            if top_class == self.class_id and result['gt_label'] == self.class_id:
                tp += 1
            if top_class != self.class_id and result['gt_label'] != self.class_id:
                tn += 1
            if top_class == self.class_id and result['gt_label'] != self.class_id:
                fp += 1
            if top_class != self.class_id and result['gt_label'] == self.class_id:
                fn += 1
                fns.append(top_class)
            
        acc = (tp + tn) / (tp + tn + fp + fn)
        if tp + fp == 0:
            prec = None
        else:
            prec = tp / (tp + fp)
        
        if tp + fn == 0:
            rec = None
        else:
            rec = tp / (tp + fn)
        
        if prec is not None and rec is not None:
            f1 = 2 * prec * rec / (prec + rec)
            f2 = (1 + 2 ** 2) * prec * rec / (2 ** 2 * prec + rec)
        else:
            f1 = None
            f2 = None
                
        return {
            str(self.class_id) + '/accuracy': acc, str(self.class_id) + '/precision': prec,
            str(self.class_id) + '/recall': rec, str(self.class_id) + '/f1': f1,
            str(self.class_id) + '/f2': f2, str(self.class_id) + '/tp': tp,
            str(self.class_id) + '/tn': tn, str(self.class_id) + '/fp': fp,
            str(self.class_id) + '/fn': fn,
        }
