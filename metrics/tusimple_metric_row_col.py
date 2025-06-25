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
class TuSimpleMetricRowCol(BaseMetric):
    def __init__(self, threshold: float = 0.6, delta_x: float = 30, positive_threshold : float = 0.75, H=384, W=640, n_points=72, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold 
        self.H = H
        self.W = W
        self.poly_dict_losses = {"LineIoULoss" :(LOSSES.build(dict(type="LineIoULoss")), 1.0)}
        self.matcher = HungarianMatcher(self.poly_dict_losses, {})
        self.delta_x = delta_x
        self.positive_threshold = positive_threshold  # Percentage of predicted points need to be less than delta threshold to be considered as a positive
        self.n_points = n_points
    def process(self, data_batch, data_samples):
        polys, class_logits = data_samples["predictions"]
        targets = data_batch["targets"]       
        B = polys.shape[0]
        fp_batch = 0
        fn_batch = 0
        tp_batch = 0
        gt_batch = 0
        preds_batch = 0
        for i in range(B):
            good_polys = polys[i, class_logits[i]>self.threshold, :]
            if(targets[i].shape[-1] > 0):
                matches = self.matcher(good_polys.unsqueeze(0), class_logits[i].unsqueeze(0), targets[i].unsqueeze(0))
                tp, fn, fp, preds, gts = self.calc_metric(good_polys[:, :].cpu().numpy(), data_batch["key_points"][i], data_batch["ori_shapes"][i], matches)
            else:
                fp = good_polys.shape[1]
                tp = 0
                fn = 0
                gts = 0
                preds = good_polys.shape[1]
            tp_batch += tp
            fn_batch += fn
            fp_batch += fp
            preds_batch += preds
            gt_batch += gts
        self.results.append([fp_batch, fn_batch, tp_batch, preds_batch, gt_batch])
    
    def calc_metric(self, good_polys, key_points, ori_shape, matches):
        tp = 0
        fn = 0
        fp = 0
        Is = matches[0][0]
        Js = matches[0][1]
        for i, j in zip(Is, Js):
            x = good_polys[i, :-2]
            y_max, y_min = good_polys[i, -2:]
            x_t = np.array(key_points[j]['x'])/ori_shape[1]
            y_t = np.array(key_points[j]['y'])/ori_shape[0]
            x_s = self.get_x_at_y(x, y_max, y_min, y_t)
            x_good = self.get_total_good_points(x_s, ori_shape[1], x_t)
            if(x_good/x_t.shape[0] > self.positive_threshold):
                tp += 1
            else:
                fp += 1
        gts = len(key_points)
        matches = matches[0][0].shape[0]
        fn += gts - matches
        preds = good_polys.shape[0]
        fp += preds - matches
        return tp, fn, fp, preds, gts

    def get_x_at_y(self, x, y_max, y_min, y_t):
        y_factors = ((y_t - y_min)/(y_max - y_min))*self.n_points
        x_at_y = np.interp(y_factors, np.arange(self.n_points), x)
        return x_at_y
        

    
    def get_total_good_points(self, x, W, x_t):
        return np.sum(np.abs(x_t - x)*W< self.delta_x)
    
    def compute_metrics(self, results):
        fps = sum([result[0] for result in results])
        fns = sum([result[1] for result in results])
        tps = sum([result[2] for result in results])
        preds = sum([result[3] for result in results])
        gts = sum([result[4] for result in results])
        
        FNR = fns/gts
        FPR = fps/max(preds, 1)
        P = tps/max(tps + fps, 1)
        R = tps/max(tps + fns, 1)
        F1 = 2*P*R/max(P+ R, 1)

        return {"TuSimple FNR": FNR, "TuSimple FPR": FPR, "TuSimple F1": F1}
