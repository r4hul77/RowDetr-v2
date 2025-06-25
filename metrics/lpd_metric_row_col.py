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
import statistics


@METRICS.register_module()
class LPDMetricRowCol(BaseMetric):
    def __init__(self, threshold: float = 0.6, H=512, W=512, n_points=72, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold 
        self.H = H
        self.W = W
        self.poly_dict_losses = {"LineIoULoss" :(LOSSES.build(dict(type="LineIoULoss")), 1.0)}
        self.matcher = HungarianMatcher(self.poly_dict_losses, {})
        self.n_points = n_points
    def process(self, data_batch, data_samples):
        polys, class_logits = data_samples["predictions"]
        targets = data_batch["targets"]       
        B = polys.shape[0]
        LPDs = []
        for i in range(B):
            good_polys = polys[i, class_logits[i]>self.threshold, :]
            matches = self.matcher(good_polys.unsqueeze(0), class_logits[i].unsqueeze(0), targets[i].unsqueeze(0))
            lpd = self.calc_metric(good_polys[:, :].cpu().numpy(), data_batch["key_points"][i], data_batch["ori_shapes"][i], matches)
            LPDs.extend(lpd)
                
        self.results.append([statistics.mean(LPDs), len(LPDs)])
    
    def get_x_at_y(self, x, y_max, y_min, y_t):
        y_factors = ((y_t - y_min)/(y_max - y_min))*self.n_points
        x_at_y = np.interp(y_factors, np.arange(self.n_points), x)
        return x_at_y
       
    def calc_metric(self, good_polys, key_points, ori_shape, matches):
        Is = matches[0][0]
        Js = matches[0][1]
        lpds = []
        ego_idxs = self.find_ego_idx(key_points, ori_shape)
        for i, j in zip(Is, Js):
            if(j not in ego_idxs):
                continue
            x = good_polys[i, :-2]
            y_max, y_min = good_polys[i, -2:]
            x_t = np.array(key_points[j]['x'])/ori_shape[1]
            y_t = np.array(key_points[j]['y'])/ori_shape[0]
            x_s = self.get_x_at_y(x, y_max, y_min, y_t)
            lpd = self.calculate_lpd(x_s, x_t, np.min(y_t), np.max(y_t), ori_shape)
            lpds.append(lpd)
        if(len(lpds) == 0):
            return [1, 1]
        return lpds
    
    def find_ego_idx(self, key_points, ori_shape):
        dist_x = []
        idxs = []
        for idx, key_point in enumerate(key_points):
            p_y = np.polyfit(key_point['y'], key_point['x'], 1)
            x_min = np.polyval(p_y, ori_shape[0])
            dist_x.append(x_min-0.5*ori_shape[1])
            idxs.append(idx)
        dist_x = np.array(dist_x)
        idxs = np.array(idxs)
        idxs_l = idxs[dist_x < 0]
        xs_l = dist_x[dist_x < 0]
        
        idxs_r = idxs[dist_x >= 0]
        xs_r = dist_x[dist_x >= 0]
        if(len(xs_l) == 0):
            return []
        if(len(xs_r) == 0):
            return []
        return idxs_l[np.argmax(xs_l)], idxs_r[np.argmin(xs_r)]
        
        
    
    def solve_lambdas(self, y, y_t):
        roots = []
        for i in range(y_t.shape[0]):
            y_fixed = np.copy(y)
            y_fixed[-1] -= y_t[i]
            try:           
                roots_i = np.roots(y_fixed)
            except ValueError:
                roots.append(np.array([-1.]))
                continue
            # eliminate roots which are not in the range of [0, 1]
            # eliminate complex roots
            roots_i = np.real(roots_i[np.isreal(roots_i)])
            roots_i = roots_i[np.logical_and(roots_i >= -0.5, roots_i <= 1.5)]
            # if there are multiple roots choose the lowest value
            if(roots_i.shape[0]!=1):
                roots.append(np.array([-1.]))
            else:
                roots.append(roots_i)
        return np.array(roots)
    
    def calculate_lpd(self, x, x_t, y_min, y_max, ori_shape):
        delta = np.sum(np.abs(x - x_t)*ori_shape[1])
        #delta += ori_shape[0]*np.sum(lambdas == -1)
        lpd = (delta/max((y_max - y_min)*ori_shape[0], 1))
        return lpd    
    
    def compute_metrics(self, results):
        lpds_total =  sum([result[0]*result[1] for result in results])
        total = sum([result[1] for result in results])

        return {"Mean LPD": lpds_total/total, "Total LPD": lpds_total}
