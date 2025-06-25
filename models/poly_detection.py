from typing import Dict
from mmengine.optim import OptimWrapper
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from timm import create_model
from mmengine.model import BaseModel
from models.poly_head import PolyHead, PolyCNNHead
from models.decoders import *
import numpy as np
import cv2
from models.matcher import HungarianMatcher
from models.loss_manager import LossManager
from torch.quantization import QuantStub, DeQuantStub
from models.loss_register import HEADS

@MODELS.register_module('PolyNet', force=True)
class PolyNet(BaseModel):
    
    def __init__(self, backbone, n_poly, n_degree, data_preprocessor, polylossWithCoeffs, classLossesWithCoeffs, viz=None, frozen=True, quant=True, head="PolyHead") -> None:
        
        super().__init__(data_preprocessor)
        
        self.backbone =  create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        out = self.backbone(torch.randn(1, 3, 256, 256))
        self.poly_head = HEADS.build(dict(type=head, n_poly=n_poly, n_degree=n_degree, input_dim=out.shape[-1]))
        self.viz = viz
        self.loss = LossManager(polylossWithCoeffs, classLossesWithCoeffs)
        self.n_poly = n_poly
        self.n_degree = n_degree
        self.quantize = quant
        if self.quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
    
    def train_step(self, data, optim_wrapper):
        inputs_dict = self.data_preprocessor(data, True)
        with torch.set_grad_enabled(True):
            pred_polys, class_logits = self.forward(inputs_dict["images"])
            loss, loss_dict = self.loss(pred_polys, class_logits, inputs_dict["targets"])
            optim_wrapper.update_params(loss)

        return loss_dict # For Reporting

    def forward_train(self, x):
        if self.quantize:
            x = self.quant(x)
        polys, class_logits = self.poly_head(self.backbone(x))
        if self.quantize:
            polys = self.dequant(polys)
            class_logits = self.dequant(class_logits)
        return polys, class_logits

    def val_step(self, data):
        inputs_dict = self.data_preprocessor(data, True)
        pred_polys, class_logits = self.forward_train(inputs_dict["images"])
        report = {}
        with torch.no_grad():
            _, loss_dict = self.loss(pred_polys, class_logits, inputs_dict["targets"])
        report.update(loss_dict)
        for key, value in report.items():
            report[key] = value.item()
        report["predictions"] = [pred_polys.cpu().detach().clone(), class_logits.cpu().detach().clone()]
        return report

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        polys, class_logits = self.poly_head(self.backbone(x))
        if self.quantize:
            polys = self.dequant(polys)
            class_logits = self.dequant(class_logits)
        return polys, class_logits
        


class PoolingUpScale(nn.Module):
    def __init__(self, in_channels, out_channels, pooling='max'):
        super().__init__()
          
        
        self.upsample = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        B, _, _, _ = x.shape
        y = x.view(B, -1).contiguous()
        x = y.max(dim=-1).values
        x = self.upsample(x)
        return x.unsqueeze(1)
class MaxFpnNet(torch.nn.Module):
    
    def __init__(self, backbone, upsample, skip_pos, frozen=True, example_in=None):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=True, features_only=True)
        example_transformed = self.backbone(example_in)
        for param in self.backbone.parameters():
            param.requires_grad = not frozen
        self.channels = self.backbone.feature_info.channels()
        self.max_sample = self.channels[-1]*upsample
        self.skip_pos = skip_pos
        
        
        self.nn_s = torch.nn.ModuleList()
        for i in range(self.skip_pos, len(self.channels)):
            self.nn_s.append(PoolingUpScale(self.channels[i], self.max_sample))
        # Create parameters for upsampling by initializing them to 1/no_channels this way we can fuse them
        print("nn_s created")
        self.num_feats_chs = len(self.channels) - self.skip_pos 
        self.alphas = nn.Parameter(torch.ones(self.num_feats_chs)*1/self.num_feats_chs, requires_grad=True)

    def forward(self, x):
        feats = self.backbone(x)
        out = [self.alphas[i]*nn_s(feats[i+self.skip_pos]) for i, nn_s in enumerate(self.nn_s)]
        out = torch.cat(out, dim=1)
        out = out.sum(dim=1)
        return out

@MODELS.register_module('PolyCNN', force=True)
class PolyCNN(PolyNet):
    
    def __init__(self, backbone, n_poly, n_degree, data_preprocessor, skip_pos, upsample, polylossesWithCoeffs, classlossesWithCoeffs, viz=None, frozen=True, img_size=(512, 512)) -> None:
        super(PolyNet, self).__init__(data_preprocessor)
        self.backbone = create_model(backbone, pretrained=True, features_only=True)
        out = self.backbone(torch.randn(1, 3, img_size[-2], img_size[-1]))
        self.poly_head = PolyCNNHead(n_poly=n_poly, n_degree=n_degree, input_dim=out[-1].shape)
        self.viz = viz
        self.loss = LossManager(polylossesWithCoeffs=polylossesWithCoeffs, classlossesWithCoeffs=classlossesWithCoeffs)
    
    def forward(self, x):
        return self.poly_head(self.backbone(x)[-1])