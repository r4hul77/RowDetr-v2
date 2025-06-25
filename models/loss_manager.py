import numpy as np
import torch.nn as nn
from models.matcher import HungarianMatcher
import torch
from torch.nn import functional as F
from models.losses import LOSSES
from models.losses import *
class LossManager(nn.Module):
    
    def __init__(self, polyLossesWithCoeffs, classLossesWithCoeffs) -> None: #lossesWithCoeffs is a dictionary containing the loss functions and their coefficients
        super().__init__()
        self.polyLossesWithCoeffs = LossManager.build_losses(polyLossesWithCoeffs)
        
        self.classLossesWithCoeffs = LossManager.build_losses(classLossesWithCoeffs)
        self.matcher = HungarianMatcher(self.polyLossesWithCoeffs, self.classLossesWithCoeffs)
        self.IouAwareLoss = IOUAwareLoss()
    
    def forward(self, pred_polys, class_logits, targets, enc_polys=None, enc_class_logits=None, dn_polys=None, dn_logits=None, dn_queries=None):
        idxs = self.matcher(pred_polys, class_logits, targets)
        losses_dict = {}
        total_loss = torch.tensor(0.).to(pred_polys[0].device)
        iou_loss = torch.tensor(0.).to(pred_polys[0].device)
        for name, _ in self.polyLossesWithCoeffs.items():
            losses_dict[name] = torch.tensor(0.).to(pred_polys[0].device)
        for name, _ in self.classLossesWithCoeffs.items():
            losses_dict[name] = torch.tensor(0.).to(pred_polys[0].device)
        for i, matches in enumerate(idxs):
            cls_tgts = torch.zeros_like(class_logits[i])
            cls_tgts[matches[0]] = 1.
            iou = torch.zeros([pred_polys[i].shape[0]]).to(pred_polys[i].device)
            for name, loss_coeff in self.polyLossesWithCoeffs.items():
                if(torch.numel(targets[i]) == 0):
                    continue
                loss, coeff = loss_coeff
                loss_now = loss(pred_polys[i, matches[0], :, :].unsqueeze(0), targets[i][matches[1], :, :].unsqueeze(0))
                iou = loss(pred_polys[i, :, :, :].unsqueeze(0), targets[i].unsqueeze(1)).min(dim=0).values
                iou[matches[0]] = loss_now.squeeze(0)
                losses_dict[name] += loss_now.squeeze().mean()
                total_loss += coeff*loss_now.squeeze().mean()
            for name, loss_coeff in self.classLossesWithCoeffs.items():
                loss, coeff = loss_coeff
                loss_now = loss(class_logits[i], cls_tgts)
                losses_dict[name] += loss_now
                total_loss += coeff*loss_now
            #iou_loss += self.IouAwareLoss(class_logits[i], cls_tgts, iou)
        #total_loss += iou_loss
        if(enc_polys is not None):
            for name, _ in self.polyLossesWithCoeffs.items():
                losses_dict[f"{name}_enc"] = torch.tensor(0.).to(pred_polys[0].device)
            for name, _ in self.classLossesWithCoeffs.items():
                losses_dict[f"{name}_enc"] = torch.tensor(0.).to(pred_polys[0].device)
            idxs = self.matcher(enc_polys, enc_class_logits, targets)
            iou = torch.zeros([pred_polys[i].shape[0]]).to(pred_polys[i].device).double()
            iou_enc = torch.tensor(0.).to(pred_polys[i].device)
            for i, matches in enumerate(idxs):
                
                cls_tgts = torch.zeros_like(enc_class_logits[i])
                cls_tgts[matches[0]] = 1.
                if(enc_polys is not None):
                    for name, loss_coeff in self.polyLossesWithCoeffs.items():
                        if(torch.numel(targets[i]) == 0):                            
                            continue
                        loss, coeff = loss_coeff
                        
                        loss_now = loss(enc_polys[i, matches[0], :, :].unsqueeze(0), targets[i][matches[1], :, :].unsqueeze(0))
                        iou[matches[0]] = loss_now.squeeze(0)
                        
                        losses_dict[f"{name}_enc"] += loss_now.squeeze().mean()
                        total_loss += coeff*loss_now.squeeze().mean()
                    for name, loss_coeff in self.classLossesWithCoeffs.items():
                        loss, coeff = loss_coeff
                        loss_now = loss(enc_class_logits[i], cls_tgts)
                        losses_dict[f"{name}_enc"] += loss_now
                        total_loss += coeff*loss_now
                    iou_enc += self.IouAwareLoss(class_logits[i], cls_tgts, iou)
            losses_dict["iou_enc_loss"] = iou_enc
            total_loss += iou_enc
        #losses_dict["iou_loss"] = iou_loss
        # total_loss /= class_logits.shape[0]
        
        if(dn_polys is not None):
            for name, loss_coeff in self.polyLossesWithCoeffs.items():
                loss, coeff = loss_coeff
                loss_now= torch.mul(dn_queries['tgt_logits'], loss(dn_polys, dn_queries['tgt_polys'])).mean()
                losses_dict[f"dn_{name}"] = loss_now
                total_loss += coeff*loss_now
            for name, loss_coeff in self.classLossesWithCoeffs.items():
                loss, coeff = loss_coeff
                loss_now = loss(dn_logits, dn_queries['tgt_logits'])
                losses_dict[f"dn_{name}"] = loss_now
                total_loss += coeff*loss_now
        losses_dict["total_loss"] = total_loss
        return total_loss, losses_dict
    
    def build_losses(lossesWithCoeffs):
        losses = {}
        for loss, coeff in lossesWithCoeffs:
            losses[loss["type"]] = (LOSSES.build(loss), coeff)
        return losses
    
    