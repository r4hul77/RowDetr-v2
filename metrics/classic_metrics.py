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
import math 
@METRICS.register_module()
class ClassicMetrics(BaseMetric):
    
    def __init__(self, threshold: float = 0.5, judging_loss_name: str = "PolyLoss", H= 512, W = 512, area_sides = range(5, 40, 5), area=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.poly_dict_losses = {judging_loss_name :(LOSSES.build(dict(type=judging_loss_name, points=100)), 1.0)}
        self.matcher = HungarianMatcher(self.poly_dict_losses, {}) # This is so that we can calculate the accuracy of the model
        self.judging_loss_name = judging_loss_name
        self.distances_container = SortedList()
        self.false_positives = 0
        self.false_negs = 0
        self.H = H
        self.W = W
        self.area_sides = area_sides
        if area == True:
            self.multiplier = math.sqrt(self.H*self.W)
        else:
            self.multiplier = self.H*self.W

    def process(self, data_batch, data_samples_list):
        data_samples = data_samples_list[0]
        polys, class_logits = data_samples["predictions"]
        targets = data_batch["targets"]       
        B = polys.shape[0]
        distance_container = SortedList()
        false_positives = 0
        false_negs = 0
        for i in range(B):
            good_polys = polys[i, class_logits[i]>self.threshold, :, :].unsqueeze(0)
            matches = self.matcher(good_polys, class_logits[i].unsqueeze(0), targets[i].unsqueeze(0))
            if(targets[i].numel() != 0):
                if self.judging_loss_name == "PointWiseIOULoss":
                    distance = self.poly_dict_losses[self.judging_loss_name][0](good_polys[0, matches[0][0], :, :].unsqueeze(0), targets[i][matches[0][1], :, :].unsqueeze(0), self.H, self.W).squeeze(0)
                else:
                    distance = torch.clip(self.poly_dict_losses[self.judging_loss_name][0](good_polys[0, matches[0][0], :, :].unsqueeze(0), targets[i][matches[0][1], :, :].unsqueeze(0)), 0, 1).squeeze(0)*self.multiplier
            else:
                distance = torch.ones(good_polys.shape[1])*self.multiplier
            distance_container.update(distance.tolist())
            
            misses = np.abs(good_polys.shape[1] - targets[i].shape[0])
            if(good_polys.shape[1] > targets[i].shape[0]):
                false_positives += misses
            else:
                false_negs += misses
        self.results.append([distance_container, false_positives, false_negs])

    def compute_metrics(self, results):
        report = {
        
        }
        self.distances_container = SortedList()
        self.false_positives = 0
        self.false_negs = 0
        
        for result in results:
            self.distances_container.update(result[0])
            self.false_positives += result[1]
            self.false_negs += result[2]
        if len(self.distances_container) == 0:
            report['Mean Area'] = self.H*self.W
            report["Mean Side"] = math.sqrt(self.H*self.W)
            for side in self.area_sides:
                report[f"F1 @ {side}"] = 0
                report[f"Precession @ {side}"] = 0
                report[f"Recall @ {side}"] = 0
            return report
        report['Mean Area'] = np.mean(self.distances_container)
        report["Mean Side"] = np.mean(np.sqrt(self.distances_container))
        self.save_histogram("histogram.png")
        prcs, recalls = [], []
        sides = []
        for side in self.area_sides:
            f1, precession, recall = self.get_metrics_at_area(side*side)
            report[f"F1 @ {side}"] = f1
            report[f"Precession @ {side}"] = precession
            prcs.append(precession)
            report[f"Recall @ {side}"] = recall
            recalls.append(recall)
            sides.append(side)
        self.save_precision_recall_curve(sides, prcs, recalls, "precision_recall_curve.png")
        return report
    
    
    def save_precision_recall_curve(self, sides, prcs, recalls, path):
        # sides as point labels, prcs as y, recalls as x
        plt.plot(recalls, prcs)
        plt.scatter(recalls, prcs)
        for i, side in enumerate(sides):
            plt.annotate(f"{side}", (recalls[i], prcs[i]))
        plt.xlabel("Recall")
        plt.ylabel("Precession")
        plt.grid()
        plt.savefig(path)
        plt.close()
            
        
    def save_histogram(self, path):
        sides = np.sqrt(self.distances_container)
        if len(sides) == 0:
            return
        plt.hist(sides, bins=100, density=True, range=(min(sides), max(sides)))
        plt.grid()
        plt.savefig(path)
        plt.close()    
    
    def get_metrics_at_area(self, area):
        true_postives = self.distances_container.bisect(area)
        false_negs = self.false_negs
        false_positives = max(self.false_positives + len(self.distances_container) - true_postives, 1)
        precession = true_postives/(true_postives + false_positives)
        recall = true_postives/max((true_postives + false_negs), 1)
        f1 = 2*precession*recall/max((precession + recall), 1e-5)
        return f1, precession, recall
    
    def __str__(self) -> str:
        ret_str = f"Classic Metrics: \n"
        report = self.compute()
        for key in report:
            ret_str += f"{key}: {report[key]:.4f}\n"
        return ret_str