from torch import nn
import torch
import numpy as np
import unittest
from mmengine.registry import Registry
from models.loss_register import LOSSES
# Create Polynomial Loss

@LOSSES.register_module()
class AbsLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, target):
        return torch.abs(pred - target).sum(dim=-1).sum(dim=-1)
    
@LOSSES.register_module()
class AbsRowColLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, target):
        return torch.abs(pred - target).sum(dim=-1)

@LOSSES.register_module()
class PolyLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()   
    
    def forward(self, pred, target):
        #pred shape [1, n_poly, 2, n_deg+1]
        #target shape [2, n_deg+1]
        deg = pred.shape[-1] - 1
        err = pred - target
        t = deg*2 + 1
        loss_x = 0
        loss_y = 0
        for i in range(0, deg+1):
            loss_x_now = 0
            loss_y_now = 0
            for j in range(0, deg+1):
                loss_x_now += err[:, :, 0, j]/(t-j)
                loss_y_now += err[:, :, 1, j]/(t-j)
            loss_x += loss_x_now*err[:, :, 0, i]
            loss_y += loss_y_now*err[:, :, 1, i]
            t -= 1
        area_loss = loss_x + loss_y
        return area_loss
@LOSSES.register_module()
class PolyOptLoss(nn.Module):
    def __init__(self, H=1.0, W=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.H = H
        self.W = W
    
    def forward(self, pred, target):
        # Add Deg to the target

        deg2Add = pred.shape[-1] - target.shape[-1]
        deg = pred.shape[-1] - 1
        # Add Zeros equal to the degree2Add
        target = torch.cat([torch.zeros_like(target[:, :, :, 0]).unsqueeze(-1).repeat(1, 1, 1, deg2Add), target], dim=-1)
        err = pred - target
        factors = torch.arange(2*deg+1, deg, -1).view(-1, 1).to(err.device) - torch.arange(deg+1).view(1, -1).to(err.device)
        err_fac = (err.unsqueeze(-2) / factors.T).sum(dim=-1)
        loss = torch.einsum("bpad, bpad -> bp", err_fac, err)
        return loss
    
@LOSSES.register_module()
class IOUAwareLoss(nn.Module):
    
    def __init__(self, k= 0.25, alpha= 0, gamma= 2) -> None:
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, targets, ious):
        ious_fixed = torch.abs(1 - ious).clamp(max=1.).detach()
        p = pred.sigmoid().detach()
        iou_aware_score = p**self.k*(1-ious_fixed).unsqueeze(-1)**(1-self.k)
        iou_aware_score = iou_aware_score.clamp(min=0.01)        
        target_score = targets*iou_aware_score
        weight = (1 - self.alpha)*p.pow(self.gamma)*(1-targets) + targets
        loss = torch.nn.functional.binary_cross_entropy_with_logits(iou_aware_score, target_score, weight=weight)
        return loss.mean()

@LOSSES.register_module()
class PxPyLoss(nn.Module):
    
    def __init__(self, px_losses, py_losses, px_coeffs, py_coeffs) -> None:
        super().__init__()
        self.px_losses = [(LOSSES.build({"type":loss}), coeffs) for loss, coeffs in zip(px_losses, px_coeffs)]
        self.py_losses = [(LOSSES.build({"type":loss}), coeffs) for loss, coeffs in zip(py_losses, py_coeffs)]

    def forward(self, pred, targets):
        loss_x = torch.zeros(targets.shape[0], pred.shape[1]).to(pred.device)
        loss_y = torch.zeros(targets.shape[0], pred.shape[1]).to(pred.device)
        for loss, coef in self.px_losses:
            loss_x += loss(pred[:, :, 0, :].unsqueeze(2), targets[:, :, 0, :].unsqueeze(2))*coef
        for loss, coef in self.py_losses:
            loss_y += loss(pred[:, :, 1, :].unsqueeze(2), targets[:, :, 1, :].unsqueeze(2))*coef
        return loss_x + loss_y
    

   
@LOSSES.register_module()
class BruteForcePolyLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, targets):
        lambdaz = torch.linspace(0, 1, 100).to(pred.device)
        lamdas = torch.stack([lambdaz**3, lambdaz**2, lambdaz, torch.ones_like(lambdaz)], dim=-1)
        
        preds = (pred*lamdas.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        targets = (targets*lamdas.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        preds[preds==0] = 1e-6
        
        loss = 0.01*((1/preds - targets)**2).sum(dim=-1)
        return loss


@LOSSES.register_module()
class PointWiseLoss(nn.Module):
    def __init__(self, points=100) -> None:
        super().__init__()
        self.points = int(points)
    
    def forward(self, pred, target):
        deg = pred.shape[-1] - 1
        #Create Lambdas
        lambdaz = torch.linspace(0, 1, self.points).to(pred.device)
        lamdas = torch.stack([lambdaz**d for d in range(deg, -1, -1)], dim=0)
        #Pred Polys
        pred_points = (pred@lamdas)
        target_points = (target.float()@lamdas)
        #Calculate the loss
        loss = ((pred_points - target_points)**2).sum(dim=-2).sum(dim=-1)*lambdaz[1]
        return loss


@LOSSES.register_module()
class PointWiseIOULoss(nn.Module):
    def __init__(self, points=100) -> None:
        super().__init__()
        self.points = int(points)
    
    def forward(self, pred, target, H=640, W=384):
        deg = pred.shape[-1] - 1
        #Create Lambdas
        lambdaz = torch.linspace(0, 1, self.points).to(pred.device)
        lamdas = torch.stack([lambdaz**d for d in range(deg, -1, -1)], dim=0)
        #Pred Polys
        pred_points = (pred@lamdas)
        target_points = (target.float()@lamdas)
        #Calculate the loss
        Pr, _, D, P = pred_points.shape
        _, T, _, _ = target_points.shape
        hw = torch.tensor([H, W]).to(pred.device).reshape(1, 1, 2, 1).repeat(Pr, T, 1, P)
        loss = torch.sqrt(((hw*(pred_points - target_points))**2).sum(dim=-2)).sum(dim=-1)*lambdaz[1]
        return loss**2
        

@LOSSES.register_module()
class PointWiseOneOverLoss(nn.Module):
    
    def __init__(self, points=10) -> None:
        super().__init__()
        self.px = PointWiseLoss()
        self.points = points

    def forward(self, pred, target):
        deg = pred.shape[-1] - 1
        #Create Lambdas
        lambdaz = torch.linspace(0, 1, self.points).to(pred.device)
        lamdas = torch.stack([lambdaz**d for d in range(deg, -1, -1)], dim=0)
        #Pred Polys
        pred_points = (pred@lamdas)
        pred_points_y = pred_points[:, :, 1]
        pred_points_y[pred_points_y==0 ] = 1e-6
        pred_points_y = 1/pred_points[:, :, 1]
        pred_points_x = pred_points[:, :, 0]
        pred_points = torch.stack([pred_points_x, pred_points_y], dim=-2)
        target_points = (target.float()@lamdas)
        #Calculate the loss
        loss = ((pred_points - target_points)**2).sum(dim=-2).sum(dim=-1)*lambdaz[1]
        return loss        

                
@LOSSES.register_module()                
class EndPointsLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()   
    
    def forward(self, pred, target):
        #pred shape [1, n_poly, 2, n_deg+1]
        #target shape [2, n_deg+1]

        deg = pred.shape[-1] - 1
        err = (torch.absolute(pred[..., 0, -1] - target[..., 0, -1]) + torch.absolute(pred[..., 1, -1] - target[..., 1, -1]))
        end_point_loss = err
        return end_point_loss

@LOSSES.register_module()
class RegLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()   
    
    def forward(self, pred, target):
        #pred shape [1, n_poly, 2, n_deg+1]
        #target shape [T, 1, 2, n_deg+1]
        deg2Add = pred.shape[-1] - target.shape[-1]
        deg = pred.shape[-1] - 1
        # Add Zeros equal to the degree2Add
        target = torch.cat([torch.zeros_like(target[:, :, :, 0]).unsqueeze(-1).repeat(1, 1, 1, deg2Add), target], dim=-1)
        err = (pred - target)**2
        reg_loss = err.sum(dim=-1).sum(dim=-1)
        return reg_loss

    
@LOSSES.register_module()
class BCELoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCELoss()
    
    def forward(self, pred, target):
        return self.loss(pred, target)  

@LOSSES.register_module()
class BCEWithLogitsLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return self.loss(pred, target)  


@LOSSES.register_module()
class LineLocLoss(nn.Module):
    def __init__(self):
        """
        Line Location Loss as defined in the paper.
        """
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted coordinates of shape [B, Q, N+2].
            target (Tensor): Ground truth coordinates of shape [B, T, N+2].

        Returns:
            Tensor: Scalar loss value.
        """
        # Extract x_min, x_max, y_min, y_max for predictions and targets
        pred_x_max = torch.max(pred[:, :, :-2], dim=-1).values
        pred_x_min = torch.min(pred[:, :, :-2], dim=-1).values
        target_x_max = torch.max(target[:, :, :-2], dim=-1).values
        target_x_min = torch.min(target[:, :, :-2], dim=-1).values
        pred_y_min, pred_y_max = torch.min(pred[:, :, -1], pred[:, :, -2]), torch.max(pred[:, :, -1], pred[:, :, -2])
        target_y_min, target_y_max = torch.min(target[:, :, -1], target[:, :, -2]), torch.max(target[:, :, -1], target[:, :, -2])
        pred_area = torch.abs((pred_y_max - pred_y_min)) * torch.abs((pred_x_max - pred_x_min))
        target_area = torch.abs((target_y_max - target_y_min)) * torch.abs((target_x_max - target_x_min))
        intersect_x = torch.clamp(
            torch.min(pred_x_max.unsqueeze(2), target_x_max.unsqueeze(1)) -
            torch.max(pred_x_min.unsqueeze(2), target_x_min.unsqueeze(1)),
            min=0
        )  # [B, Q, T]
        intersect_y = torch.clamp(
            torch.min(pred_y_max.unsqueeze(2), target_y_max.unsqueeze(1)) -
            torch.max(pred_y_min.unsqueeze(2), target_y_min.unsqueeze(1)),
            min=0
        )  # [B, Q, T]
        intersection_area = intersect_x * intersect_y 
        union_area = pred_area + target_area - intersection_area.squeeze(-1)
        iou = torch.clamp(intersection_area.squeeze(-1) / (union_area + 1e-6), 0, 1)
        loss = 1 - iou
        return loss

@LOSSES.register_module()
class LineIoULoss(nn.Module):
    def __init__(self, epsilon=0.0078125):
        """
        LineIoU Loss as defined in the paper.

        Args:
            epsilon (float): Small value for boundary relaxation.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted coordinates of shape [B, Q, N+2].
            target (Tensor): Ground truth coordinates of shape [B, T, N+2].

        Returns:
            Tensor: Scalar loss value.
        """
        # Extract x-coordinates (N points) from predictions and targets
        pred_x = pred[..., :-2]  # Shape [B, Q, N]
        target_x = target[..., :-2]  # Shape [B, T, N]

        # Compute intersection (d_o) and union (d_u)
        d_o = torch.minimum(pred_x + self.epsilon, target_x + self.epsilon) - torch.maximum(pred_x - self.epsilon, target_x - self.epsilon)
        d_u = torch.maximum(pred_x + self.epsilon, target_x + self.epsilon) - torch.minimum(pred_x - self.epsilon, target_x - self.epsilon)

        # Sum over points and queries
        intersection = d_o.sum(dim=-1)  # Shape [B, Q, T]
        union = d_u.sum(dim=-1)  # Shape [B, Q, T]

        # Compute IoU
        iou = intersection / (union + 1e-6)  # Avoid division by zero

        # Loss: 1 - IoU
        loss = 1 - iou  # Average across batches, queries, and targets
        return loss

if __name__ == "__main__":
    
    class TestPolyLoss(unittest.TestCase):
        
        
        
        def test_poly_loss(self):
            pred = torch.rand(1, 10, 2, 4)
            pred[:, :, :, 0] = 1
            target = torch.rand(2, 1, 2, 4)
            loss1 = PolyOptLoss()
            loss2 = PointWiseLoss(points=1e7)
            val = loss1(pred, target)
            val2 = loss2(pred, target)
            assert torch.allclose(val, val2, rtol=1e-5), "Poly Losses are not equal"
            
        
    

    #run all tests
    unittest.main()
