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
import numpy as np
import cv2
from models.matcher import HungarianMatcher
from models.loss_manager import LossManager
from torch.quantization import QuantStub, DeQuantStub
from models.loss_register import HEADS
import math
from models.utils import get_activation
from models.decoders import *
import random
@MODELS.register_module('NeuRowNet', force=True)
class NeuRowNet(BaseModel):
    
    def __init__(self,
                 backbone,
                 n_poly,
                 n_degree,
                 data_preprocessor,
                 polylossWithCoeffs,
                 classLossesWithCoeffs,
                 viz=None,
                 frozen=True,
                 quant=True,
                 levels=3,
                 head_levels=2,
                 em_dim = 256,
                 head="PolyCNNHead",
                 encoder_config=None,
                 atten_type='poly',
                 size=(768, 768),
                 use_poly_act=False,
                 contrastive_denoising = True,
                 dn_grps = 16,
                 dn_queries=5,
                 sampling_points=4, # Points to sample from the polynomial len(lambdas) = sampling points, x = P_x(lambda), y = P_y(lambda)
                 num_points=3, # Points to sample from the sampled points F(G(x, y)) = num_points*(len(x))
                 num_heads=8,
                 pred_layers=3,
                 deform_levels=3,
                 atten_viz=True) -> None:
        
        super().__init__(data_preprocessor)
        self.backbone =  create_model(backbone, pretrained=True, features_only=True)
        out = self.backbone(torch.randn(1, 3, size[0], size[1]))
        self.encoder = HybridEncoder(em_dim, levels, size, strides = self.backbone.feature_info.reduction()[-levels:], channels=self.backbone.feature_info.channels(), act='relu', layers_out=head_levels)
        #self.poly_head = HEADS.build(dict(type=head, n_poly=n_poly, n_degree=n_degree, channels=self.backbone.feature_info.channels(),input_dim=[1, em_dim, out[-1].shape[-2], out[-1].shape[-1]]))    
        
        self.poly_head = HEADS.build(dict(type=head,
                                          n_poly=n_poly,
                                          n_degree=n_degree,
                                          channels=self.backbone.feature_info.channels()[levels:],
                                          strides = self.backbone.feature_info.reduction()[levels:],
                                          atten_type=atten_type,
                                          em_dim=em_dim,
                                          num_points=num_points,
                                          sampling_points=sampling_points,
                                          num_levels=head_levels,
                                          nhead=num_heads,  
                                          input_dim=[1, em_dim, out[-1].shape[-2], out[-1].shape[-1]],
                                          use_poly_act=use_poly_act,
                                          pred_layers=pred_layers,
                                          levels_in=head_levels,
                                          deform_levels=deform_levels))    
        self.viz = viz
        self.atten_viz = atten_viz
        self.loss = LossManager(polylossWithCoeffs, classLossesWithCoeffs)
        self.n_poly = n_poly
        self.n_degree = n_degree
        self.quantize = quant
        self.em_dim = em_dim
        if self.quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        self.dn = contrastive_denoising
        if contrastive_denoising:
            self.tgt_dn = nn.Embedding(2, self.em_dim, dtype=torch.float)
            self.dn_grps = dn_grps
            self.dn_queries = dn_queries
    
    def contrastive_denoising(self, targets):
        dn_batches_polys = []
        dn_polys_target = []
        dn_logits_target = []
        for batch_idx, batch_targets in enumerate(targets):
            dn_batch_polys = []
            batch_polys_target = []
            batch_logits = []
            for i in range(self.dn_grps):
                j = 0
                if(batch_targets.numel()//8<self.dn_queries):
                # Q, 8
                    j = self.dn_queries - batch_targets.numel()//8
                    dn_no_queries = torch.randn((j, 2, self.n_degree+1), device=batch_targets.device)
                    dn_batch_polys.append(dn_no_queries)
                    batch_polys_target.append(dn_no_queries)
                    batch_logits.append(torch.zeros(j, device=batch_targets.device))
                toss = torch.rand(self.dn_queries-j)
                match_queries = torch.sum(toss>0.15)
                no_match_queries = torch.sum(toss<=0.15)
                if(no_match_queries.item() > 0):
                    dn_no_queries = torch.randn((no_match_queries, 2, self.n_degree+1), device=batch_targets.device)
                    dn_batch_polys.append(dn_no_queries)
                    batch_polys_target.append(dn_no_queries)
                    batch_logits.append(torch.zeros(no_match_queries, device=batch_targets.device))
                if(match_queries.item()>0):
                    
                    dn_match_idxs = torch.multinomial(torch.ones(batch_targets.shape[0], device=batch_targets.device), match_queries, replacement=True)
                    dn_match_queries = torch.normal(mean=batch_targets[dn_match_idxs, :, :], std=0.25)
                    dn_batch_polys.append(dn_match_queries)
                    batch_polys_target.append(batch_targets[dn_match_idxs, :, :])
                    batch_logits.append(torch.ones(match_queries, device=batch_targets.device))
                
                
            dn_batches_polys.append(torch.cat(dn_batch_polys, dim=0))
            dn_polys_target.append(torch.cat(batch_polys_target, dim=0))
            dn_logits_target.append(torch.cat(batch_logits, dim=0))
        dn_ret_dict = {}
        dn_ret_dict['tgt_logits'] = torch.stack(dn_logits_target, dim=0).float()
        dn_ret_dict['tgt_polys'] = torch.stack(dn_polys_target, dim=0).float()
        dn_ret_dict['denoising_polys'] = torch.stack(dn_batches_polys).float()
        tgt_embeddings = self.tgt_dn(dn_ret_dict['tgt_logits'].long())
        dn_ret_dict['denoising_class'] = tgt_embeddings + torch.randn_like(tgt_embeddings) * 0.25
        dn_ret_dict['denoising_groups'] =[self.dn_queries]*self.dn_grps
        return dn_ret_dict
        
                
                
        

    def train_step(self, data, optim_wrapper):
        inputs_dict = self.data_preprocessor(data, True)
        with torch.set_grad_enabled(True):
            if(self.dn):
                dn_queries = self.contrastive_denoising(inputs_dict['targets'])
            pred_polys, class_logits, enc_polys, enc_logits, dn_polys, dn_logits = self._forward(inputs_dict["images"], dn_queries)
            loss, loss_dict = self.loss(pred_polys, class_logits, inputs_dict["targets"], enc_polys, enc_logits, dn_polys, dn_logits, dn_queries)
            optim_wrapper.update_params(loss)

        return loss_dict # For Reporting

    def val_step(self, data):
        inputs_dict = self.data_preprocessor(data, True)
        report = {}
        self.eval()
        with torch.no_grad():
            pred_polys, class_logits = self.forward(inputs_dict["images"], mode="predict")
            _, loss_dict = self.loss(pred_polys, class_logits, inputs_dict["targets"])
        self.train()
        report.update({"loss": loss_dict})
        for key, value in report["loss"].items():
            report["loss"][key] = value.item()
        report["predictions"] = [pred_polys.cpu().detach().clone(), class_logits.cpu().detach().clone()]
        return [report]
      
    def test_step(self, data):
        return self.val_step(data)

    def _forward(self, x, dn_queries=None):
        if self.quantize:
            x = self.quant(x)
        x = self.backbone(x)
        x = self.encoder(x)
        outs = self.poly_head(x, dn_queries=dn_queries)
        if self.quantize:
            temp = []
            for out in outs:
                temp.append(self.dequant(out))
        return outs
    
    def forward(self, images, targets=None, mode="loss",**kwargs):

        if mode == "loss":
            if(self.dn):
                dn_queries = self.contrastive_denoising(targets)
            outs = self._forward(images, dn_queries)

            polys, logits, enc_polys, enc_logits, dn_polys, dn_logits = outs
            loss, loss_dict = self.loss(polys, logits, targets, enc_polys, enc_logits, dn_polys, dn_logits, dn_queries)
            return loss_dict
        else:
            outs = self._forward(images)
            polys, logits = outs
            return polys, logits.sigmoid()
    
    def forward_viz(self, images, targets=None):
        x = self.backbone(images)
        enc_out = self.encoder(x)
        outs = self.poly_head(x)

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)




class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output
class HybridEncoder(nn.Module):
    def __init__(self,
                 em_dim:int=256,
                 levels_to_use:int=3,
                 size=(768, 768),
                 strides=[2, 4, 16, 64],
                 channels=[64, 128, 256, 512],
                 act='silu',
                 layers_out=4):
        super().__init__()
        self.encoder = TransformerEncoder(TransformerEncoderLayer(em_dim, 16, activation='relu', dim_feedforward=1024), 1)
        # This has to be changed
        self.input_proj = nn.ModuleList()
        
        self.em_dim = em_dim
        # Projector for each intersting level    
        for level in range(levels_to_use):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(channels[-level-1], em_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(em_dim)
                )
            )
    
        # top-down fpn
        self.layers_out = layers_out
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(levels_to_use - 1):
            self.lateral_convs.append(ConvNormLayer(em_dim, em_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(em_dim * 2, em_dim, round(3 * 1.0), act=act, expansion=1.0)
            )
        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(levels_to_use - 1):
            self.downsample_convs.append(
                ConvNormLayer(em_dim, em_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(em_dim * 2, em_dim, round(3 * 1.0), act=act, expansion=1.0)
            )
                
        
        self.set_pos_embed(size, strides)
    def set_pos_embed(self, size, strides):
        pos_embed = self.build_2d_sincos_position_embedding(w=size[0]//strides[-1], h=size[1]//strides[-1], embed_dim=self.em_dim)
        setattr(self, f'pos_embed', pos_embed)
    def preprocess(self, x):
        return x.flatten(2, 3).transpose(1, 2).contiguous()
    
    def postprocess(self, x, B, H, W):
        
        return x.transpose(1, 2).contiguous().view(B, -1, H, W)  
    
    def forward(self, x):
        projected_feats = [proj(x[-i-1]) for i, proj in enumerate(self.input_proj)]
        B, _, H, W = projected_feats[0].shape
        projected_feats[0] = self.postprocess(self.encoder(self.preprocess(projected_feats[0]), pos_embed=self.pos_embed.to(projected_feats[0].device)), B, H, W)
        
        # FPN pass
        feats = [projected_feats[0]]
        for i in range(1, len(projected_feats)):
            feat_high = feats[-1]
            feat_low  = projected_feats[i]
            feat_high = self.lateral_convs[i-1](feat_high)
            feats[-1] = feat_high
            upsampled = F.interpolate(feat_high, scale_factor=2.0, mode='nearest')
            fussed = self.fpn_blocks[i-1](torch.cat([upsampled, feat_low], dim=1))
            feats.append(fussed)
            
        outs = [feats[-1]]
        for idx in range(1, len(projected_feats)):
            feat_low  = outs[-1]
            feat_high = feats[-idx-1]
            
            downsampled = self.downsample_convs[idx-1](feat_low)
            out = self.pan_blocks[idx-1](torch.cat([downsampled, feat_high], dim=1))
            outs.append(out)
        return outs[-self.layers_out:]

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
            
        
        