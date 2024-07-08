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
    def __init__(self, class_id: int, thrs: float = 0, topk=1) -> None:
        super().__init__()
        self.class_id = class_id
        self.thrs = thrs
        self.topk = topk

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_score' in data_sample:
                pred_score = data_sample['pred_score'].cpu()
                # pred_label = pred_score.argmax(dim=0, keepdim=True)
                # if pred_score[pred_label] < self.thrs:
                #     pred_label = None

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
        fps = []
        for result in results:
            pred_scores = result['pred_score']
            topk_pred_scores, topk_classes = torch.topk(pred_scores, self.topk, dim=0)
            predicted = False
            if self.class_id in topk_classes:
                if pred_scores[self.class_id] >= self.thrs:
                    predicted = True

            if predicted and result['gt_label'] == self.class_id:
                tp += 1
            if not predicted and result['gt_label'] != self.class_id:
                tn += 1
            if predicted and result['gt_label'] != self.class_id:
                fp += 1
                fps.append(result['pred_label'])
            if not predicted and result['gt_label'] == self.class_id:
                fn += 1
                fns.append(result['pred_label'])
            
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
        else:
            f1 = None
        
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'fns': fns, 'fps': fps}
