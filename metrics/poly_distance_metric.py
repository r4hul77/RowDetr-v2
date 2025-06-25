# This Has Custom Metrics for the Model

from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS, EVALUATOR
from mmengine.evaluator import Evaluator
import numpy as np
from models.loss_register import LOSSES
import torch
from models.matcher import HungarianMatcher
from sortedcontainers import SortedList
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@METRICS.register_module()
class PolyDistanceMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, judging_loss_name: str = "PolyLoss", H=512, W=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold 
        self.H = H
        self.W = W
        self.poly_dict_losses = {judging_loss_name :(LOSSES.build(dict(type=judging_loss_name, points=100)), 1.0)}
        self.matcher = HungarianMatcher(self.poly_dict_losses, {}) # This is so that we can calculate the accuracy of the model
        self.judging_loss_name = judging_loss_name
        self.reset()

    def reset(self):
        self.results = []

    def process(self, data_batch, data_sample_list):
        


        data_samples = data_sample_list[0]
        results_now = data_samples["loss"]
        polys, class_logits = data_samples["predictions"]
        targets = data_batch["targets"]       
        matches_for_acc = self.matcher(polys, class_logits, targets)
        correct = 0
        for i, match_b in enumerate(matches_for_acc):
            logits = np.zeros_like(class_logits[i])
            logits[match_b[0]] = 1.
            correct += np.logical_not(np.logical_xor(logits, class_logits[i]>self.threshold)).sum()
        distance = 0
        targets_cnt = 0
        B = polys.shape[0]
        false_neg = 0
        false_postives = 0
        for i in range(B):
            good_polys = polys[i, class_logits[i]>self.threshold, :, :].unsqueeze(0)
            if(targets[i].numel() != 0):
                matches = self.matcher(good_polys, class_logits[i].unsqueeze(0), targets[i].unsqueeze(0))
                distance += torch.clip(self.poly_dict_losses[self.judging_loss_name][0](good_polys[0, matches[0][0], :, :].unsqueeze(0), targets[i][matches[0][1], :, :].unsqueeze(0)), 0, 1).sum().item()
            misses = np.abs(good_polys.shape[1] - targets[i].shape[0])
            targets_cnt += targets[i].shape[0]
            if(good_polys.shape[1] > targets[i].shape[0]):
                false_postives += misses
            else:
                false_neg += misses
            distance += misses*2
        results_now['false_neg'] = false_neg
        results_now['false_postives'] = false_postives     
        results_now['count'] = class_logits.shape[0]*class_logits.shape[1]
        results_now['total_crct'] = correct
        results_now['distance'] = distance
        results_now['targets_cnt'] = targets_cnt
        self.results.append(results_now)

    def compute_metrics(self, results):
        report = {
        
        }

        for key in results[0].keys():
            if key[-4:].lower() == "loss":
                report[f"{key}_val"] = np.mean([result[key] for result in self.results])
        
        total = sum([result['count'] for result in self.results])
        correct = sum([result['total_crct'] for result in self.results])
        report['accuracy'] = correct / total
        distance = sum([result['distance'] for result in self.results])
        targets_cnt = sum([result['targets_cnt'] for result in self.results])
        report['Mean PolyDistance'] = distance / targets_cnt
        report['false_neg_percent'] = sum([result['false_neg'] for result in self.results])/targets_cnt*100
        report['false_postives_percent'] = sum([result['false_postives'] for result in self.results])/targets_cnt*100
        return report
            
        
        
    def __str__(self) -> str:
        return f"PolyDistanceMetric: {self.compute():.4f}"


@EVALUATOR.register_module()
class PolyDistanceEvaluator(Evaluator):
    
    def __init__(self, metrics: Sequence[BaseMetric]):
        super().__init__(metrics)
    
    def process(self, data_samples: Sequence, data_batch: List):
        for metric in self.metrics:
            metric.process(data_batch, data_samples)
    