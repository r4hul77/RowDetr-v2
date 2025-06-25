import torch.nn.functional as F
import torch.nn as nn
import torch
from models.loss_register import HEADS

@HEADS.register_module()
class PolyHead(nn.Module):
    
    def __init__(self, n_poly, n_degree, input_dim) -> None:
        super().__init__()
        self.n_poly = n_poly
        self.n_degree = n_degree
        self.output_dim = 2 * n_poly * (n_degree+1)
        
        self.proj = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.ELU(),
                                        nn.Linear(input_dim//2, self.output_dim),
                                        )
        self.class_head = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.ReLU(),
                                        nn.Linear(input_dim//2, self.n_poly))

    def forward(self, x):
        class_logits = nn.functional.sigmoid(self.class_head(x))
        x = self.proj(x)
        
        return x.view(-1, self.n_poly, 2, self.n_degree+1), class_logits

@HEADS.register_module()
class NeuRowNetPolyHead(nn.Module):
    
    def __init__(self, n_poly, n_degree, channels, input_dim, **kwargs) -> None:
        super().__init__()
        self.n_poly = n_poly
        self.n_degree = n_degree
        self.output_dim = 2 * n_poly * (n_degree+1)
        self.proj = nn.Sequential(nn.Linear(input_dim[1]*3, input_dim[1]//2), nn.ELU(),
                                        nn.Linear(input_dim[1]//2, self.output_dim),
                                        )
        self.class_head = nn.Sequential(nn.Linear(input_dim[1]*3, input_dim[1]//2), nn.ReLU(),
                                        nn.Linear(input_dim[1]//2, self.n_poly))
        self.adaptive_avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        avged = []
        for lev in x:
            avged.append(self.adaptive_avg(lev).squeeze(-1).squeeze(-1))
        
        x = torch.cat(avged, dim=-1)
        
        class_logits = nn.functional.sigmoid(self.class_head(x))
        x = self.proj(x)
        
        return x.view(-1, self.n_poly, 2, self.n_degree+1), class_logits

class ActY(nn.Module):
    
    def __init__(self):
        super().__init__()

    
    def forward(self, x):
        y = x.reshape(-1, 10, 4)
        z = y.clone()
        y[:, :, 3] = 1 + torch.exp(z[:, :, 3])
        y[:,:, 1] = 3*z[:, :, 0]**2 - 2*z[:, :, 0]*z[:, :, 2] + torch.exp(z[:, :, 1])
        d = 1 + torch.exp(y[:, :, 3])
        b = 3*y[:, :, 0]**2 + 2*y[:, :, 0]*y[:, :, 2] + torch.exp(y[:, :, 1])
        y = torch.stack([y[:, :, 0], b, y[:, :, 2], d], dim=-1)
        return y

@HEADS.register_module()
class PolyOneOverHead(nn.Module):
    
    def __init__(self, n_poly, n_degree, input_dim) -> None:
        super().__init__()
        self.n_poly = n_poly
        self.n_degree = n_degree
        self.output_dim =  n_poly * (n_degree+1)
        self.proj_x = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.ELU(),
                                        nn.Linear(input_dim//2, self.output_dim),
                                        )

        self.proj_y = nn.Sequential(nn.Linear(input_dim, input_dim//2),
                                        nn.Linear(input_dim//2, self.output_dim), ActY())
        
        self.class_head = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.ReLU(),
                                        nn.Linear(input_dim//2, self.n_poly))

    def forward(self, x):
        class_logits = nn.functional.sigmoid(self.class_head(x))
        X = self.proj_x(x)
        Y = self.proj_y(x)
        x = torch.stack([X.view(-1, self.n_poly, self.n_degree+1), Y.view(-1, self.n_poly, self.n_degree+1)], dim=-2)
        return x, class_logits



@HEADS.register_module()
class NeuRowNetDecoder(nn.Module):
    
    def __init__(self, n_queries, em_size) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.em_size = em_size
    
    
    
@HEADS.register_module()
class PolyCNNHead(nn.Module):
    
    def __init__(self, n_poly, n_degree, input_dim) -> None:
        super().__init__()
        self.n_poly = 256
        self.n_degree = n_degree
        self.output_dim = 2 * (n_degree+1)
        C = input_dim[1]
        self.classification_head = nn.Sequential(nn.Conv2d(input_dim[1], 1, kernel_size=(3, 3), padding=1), nn.Flatten(), nn.Sigmoid())
        self.poly_head = nn.Sequential(nn.Conv2d(input_dim[1], self.output_dim*4, kernel_size=(3, 3), padding=1),
                                       nn.ELU(),
                                       nn.Conv2d(self.output_dim*4, self.output_dim, kernel_size=(3, 3), padding=1),
                                       nn.Flatten()
                                       )
        
    def forward(self, x):
        class_logits = self.classification_head(x)
        x = self.poly_head(x)
        return x.view(-1, self.n_poly, 2, self.n_degree+1), class_logits


import math

class DeformableAtt(nn.Module):
    
    def __init__(self, embed_dim, n_heads) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.n_heads = n_heads
    
    def forward(self, x, q, ref_points):
        # Q -> B, Q, C
        # X -> B, C, H, W
        # ref_points -> B, Q, N, 2
        B, Q, C = q.shape
        _, C, H, W = x.shape
        q = self.q_proj(q)
        _, _, N, _ = ref_points.shape
        sampled_points = F.grid_sample(x, ref_points)
        k = self.k_proj(sampled_points.transpose(1, -1).contiguous())
        v = self.v_proj(sampled_points.transpose(1, -1).contiguous())
        q = q.view(B, Q, 1, self.n_heads, C//self.n_heads).transpose(2, 3).contiguous()
        k = k.view(B, Q, N, self.n_heads, C//self.n_heads).transpose(2, 3).contiguous()
        v = v.view(B, Q, N, self.n_heads, C//self.n_heads).transpose(2, 3).contiguous()
        
        attn = torch.matmul(q, k.transpose(-2, -1).contiguous())/math.sqrt(C//self.n_heads)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.view(B, Q, C)
        
        return out
        
        

@HEADS.register_module()
class PolyDecoderHead(nn.Module):
    
    def __init__(self, n_poly, n_degree, input_dim) -> None:
        super().__init__()
        self.n_poly = n_poly
        self.n_degree = n_degree
        self.output_dim = 2 * (n_degree+1)
        self.query_selection_head = nn.Sequential(nn.Linear(input_dim[1], input_dim[1]*2), nn.ELU(),
                                        nn.Linear(input_dim[1]*2, 1)
                                        )
        self.ref_points_head = nn.Sequential(nn.Linear(input_dim[1], input_dim[1]*2),
                                        nn.Linear(input_dim[1]*2, self.output_dim*2),
                                        nn.Tanh()
                                        )
        self.self_attn = nn.MultiheadAttention(input_dim[1], 8, batch_first=True)
        self.def_attn = DeformableAtt(input_dim[1], 8)        
        self.drop_out1 = nn.Dropout(0.125)
        self.drop_out2 = nn.Dropout(0.125)
        self.norm1 = nn.LayerNorm(input_dim[1])
        self.norm2 = nn.LayerNorm(input_dim[1])
        self.norm3 = nn.LayerNorm(input_dim[1])
        
        self.class_head = nn.Sequential(nn.Linear(input_dim[1], input_dim[1]*2), nn.ELU(),
                                        nn.Linear(input_dim[1]*2, 1), nn.Sigmoid())
        
        self.poly_head = nn.Sequential(nn.Linear(input_dim[1], input_dim[1]*2), nn.SiLU(),
                                        nn.Linear(input_dim[1]*2, self.output_dim),
                                        )
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def preprocess_input(self, x):
        B, C, H, W = x.shape
        memory = x.flatten(2, 3).transpose(1, 2).contiguous()
        logits = self.query_selection_head(memory)
        _, idxs = torch.topk(logits, self.n_poly, dim=-2)
        querys = memory.gather(1, idxs.repeat(1, 1, C))
        ref_points = self.ref_points_head(memory)
        ref_points_intrst = ref_points.gather(1, idxs.repeat(1, 1, self.output_dim*2)).view(B, self.n_poly, self.output_dim, 2)        
        return querys, ref_points_intrst
    def forward(self, x):
        querys, ref_points = self.preprocess_input(x)
        tgt = querys    
        out1, attn = self.self_attn(querys, querys, querys)
        tgt = tgt + self.drop_out1(out1)
        tgt = self.norm1(tgt)
        out2 = self.def_attn(x, tgt, ref_points)
        tgt = tgt + self.drop_out2(out2)
        x = self.poly_head(tgt)
        class_logits = self.class_head(tgt)
        return x.view(-1, self.n_poly, 2, self.n_degree+1), class_logits.squeeze(-1)
    

        
        
        