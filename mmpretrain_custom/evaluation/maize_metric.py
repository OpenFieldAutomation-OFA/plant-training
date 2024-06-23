from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS

@METRICS.register_module()
class MaizeMetric(BaseMetric):
    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_score' in data_sample:
                pred_score = data_sample['pred_score']
                pred_label = pred_score.argmax(dim=0, keepdim=True)
                self.num_classes = pred_score.size(0)
            else:
                pred_label = data_sample['pred_label']

            self.results.append({
                'pred_label': pred_label,
                'gt_label': data_sample['gt_label'],
            })

    def compute_metrics(self, results: list) -> dict:
        correct = 0
        for result in results:
            if result['pred_label'] == 2473 and result['gt_label'] == 2473:
                correct += 1
            if result['pred_label'] != 2473 and result['gt_label'] != 2473:
                correct += 1
            
        
        return {'maize accuracy': correct / len(results) * 100}
