from models.utils import *
import torch.nn.init as init 
import math 
from models.loss_register import HEADS
from models.act_funcs import PolyAct, InvPolyAct
import copy
from collections import OrderedDict
from models.attention_mechanisms import PolyAttentionMechanismV2
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu', bias=True):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k, bias=bias) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# Understand this shite
class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=16, n_degree=3):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.lambdaz = torch.linspace(0, 1, num_points, dtype=torch.float32)
        
        self.lambdas = torch.stack([self.lambdaz ** i for i in range(n_degree, -1, -1)], 1)
        self.n_degree = n_degree
        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).view(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        # init.constant_(self.attention_weights.weight, 0)
        # init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.view(bs, Len_v, self.num_heads, self.head_dim).contiguous()

        sampling_offsets = self.sampling_offsets(query).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2).contiguous()
        attention_weights = self.attention_weights(query).view(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points).contiguous()
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points).contiguous()

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).view(
                1, 1, 1, self.num_levels, 1, 2).contiguous()
            sampling_locations = reference_points.view(
                bs, Len_q, 1, 1, self.num_points, 2
            ) + sampling_offsets / offset_normalizer.to(reference_points.device).contiguous()
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        elif reference_points.shape[-1] == 8:
            reference_polys = reference_points.view(bs, Len_q, 2, self.n_degree+1).contiguous()
            pnts = (reference_polys@(self.lambdas.t().to(reference_polys.device))).transpose(-1, -2).contiguous()
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).view(
                1, 1, 1, self.num_levels, 1, 2).contiguous()
            sampling_locations = pnts.view(
                bs, Len_q, 1, 1, self.num_points, 2
            ).contiguous() + sampling_offsets / offset_normalizer.to(pnts.device).contiguous()            
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4 or 8 if your doing polys, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output, sampling_locations, attention_weights
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="silu",
                 n_levels=4,
                 n_points=4,
                 deform_levels=2,
                cross_atten=None):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.start_idx = n_levels - deform_levels
        if not cross_atten:
            self.cross_attn = MSDeformableAttention(d_model, n_head, deform_levels, n_points)
        else:
            self.cross_attn = cross_atten
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.levels_idxs = None
        self.levels_idxs_start = None
    #     self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_normal_(self.linear1)
        init.xavier_normal_(self.linear2)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                sampling_points=None,
                sampling_levels=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if self.levels_idxs is None:
            levels = [h*w for h, w in memory_spatial_shapes]
            levels.insert(0, 0)
            self.levels_idxs = levels
            self.levels_idxs_start = sum(levels[:self.start_idx+1])
        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))
        # mask should be (n_querries + added_groups*total_masks) shape
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2, sampling_locations, attn_weights = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory[:, self.levels_idxs_start:], 
            memory_spatial_shapes[self.start_idx:], 
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt, sampling_locations, attn_weights



class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, sampling_points= 8, degree= 3, query_num=10, num_points=2,eval_idx=-1, atten_type='poly', act=nn.Identity(), inv_act=nn.Identity(),s = 0.077):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.q_size = query_num
        self.degree = degree
        lamdaz = torch.linspace(0, 1, sampling_points, dtype=torch.float32)
        self.lambdas = torch.stack([lamdaz ** i for i in range(degree, -1, -1)], 1)
        self.num_points = num_points
        self.atten = self.poly_attn if atten_type == 'poly' else self.poly_deform_attn if atten_type == 'deform' else self.poly_sampling_attn
        if atten_type == 'polyv2':
            self.atten = self.no_atten
        self.s = s
        self.sampler_head = MLP(hidden_dim, hidden_dim, 2*num_points, act='tanh', num_layers=2)
        self.act = act
        self.inv_act = inv_act
    def poly_attn(self, ref_polys_detach, levels, ref_levels, ref_polys):
        bs, qs, _ = ref_polys_detach.shape
        sampling_points_input = 2*((ref_polys_detach.view(-1, qs, 2, self.degree+1)@(self.lambdas.t().to(ref_polys.device).type(ref_polys.type()))).transpose(-1, -2).contiguous() -0.5)
        return sampling_points_input.view(bs, qs, -1, 2).contiguous()
    
    def poly_deform_attn(self, ref_polys_detach, levels, ref_levels, ref_polys, get_sampling_pnts=False):
        bs, qs, _ = ref_polys_detach.shape
        sampling_points_input = 2*((ref_polys_detach.view(-1, qs, 2, self.degree+1)@(self.lambdas.t().to(ref_polys.device).type(ref_polys.type()))).transpose(-1, -2).contiguous() -0.5)
    
    
        # Grab the value from sampling_points_input depending upon the level
        sampling_embeddings = torch.stack(
            [F.grid_sample(level, sampling_points_input.float(),
                        mode='bilinear', padding_mode='zeros',
                        align_corners=False).permute(0, 2, 3, 1)
            for level in levels], 1)
        
        B, L, Q, S, C = sampling_embeddings.shape
        sampling_embeddings = sampling_embeddings.gather(2, ref_levels.view(B, 1, Q, 1, 1).repeat(1, 1, 1, S, C).contiguous())
        _, _, Np, _ = sampling_points_input.shape
        #ref_points_input = self.s*nn.functional.tanh(self.sampler_head(sampling_embeddings).reshape(B, Q, -1, 2))
        ref_points_input = (self.s*nn.functional.tanh(self.sampler_head(sampling_embeddings).view(B, Q, Np, -1, 2).contiguous()) + sampling_points_input.unsqueeze(-2)).view(B, Q, -1, 2).contiguous()
        if get_sampling_pnts:
            return ref_points_input, sampling_points_input
        return ref_points_input
    
    def poly_sampling_attn(self, ref_polys_detach, levels, ref_levels, ref_polys):
        ref_pnts, sampling_pnts = self.poly_deform_attn(ref_polys_detach, levels, ref_levels, ref_polys, get_sampling_pnts=True)
        ref_points_input = torch.concat([ref_pnts, sampling_pnts], -2)        
        return ref_points_input
    
    def no_atten(self, ref_polys_detach, levels, ref_levels, ref_polys):
        return ref_polys_detach
        

    def forward(self,
                tgt,
                ref_polys,
                ref_levels,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                viz=False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_polys_detach = ref_polys.detach()
        levels = [memory[:, level_start_index: level_start_index + h * w, :].transpose(-1, -2).contiguous().view(-1, self.hidden_dim, h, w)
                  for (level_start_index, (h, w)) in zip(memory_level_start_index, memory_spatial_shapes)]
        ref_points_layer = []
        for i, layer in enumerate(self.layers):
            
            
            ref_points_input = self.atten(self.act(ref_polys_detach.float()), levels, ref_levels, ref_polys)
            
            query_pos_embed = query_pos_head(self.act(ref_polys.float()))

            if viz:
                ref_points_layer.append(ref_points_input)

            output, sampling_locations, attn_weights = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = bbox_head[i](output) + ref_polys_detach

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(bbox_head[i](output) + ref_points)

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox
        if viz:
            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), sampling_locations, attn_weights
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), None, None
@HEADS.register_module()
class Decoder(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 em_dim=256,
                 n_poly=25,
                 position_embed_type='sine',
                 channels=[512, 1024, 2048],
                 strides=[8, 16, 32],
                 num_levels=3,
                 nhead=8,
                 num_decoder_layers=1,
                 dim_feedforward=1024,
                 dropout=0.0,
                 activation="relu",
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 n_degree=3, 
                 aux_loss=False,
                 sampling_points=4,
                 atten_type='poly', # poly or deform or poly +deform
                 num_points=2,
                 s = 0.14,
                 pred_layers=2, # Last layers of the feature map to give queries [3, 2, 1, 0]
                 deform_levels=2, # Layers to do cross attention [0, 1, 2]
                 levels_in = 4, # Number of levels in the feature map [0, 1, 2, 3]
                 use_poly_act=False,
                 **kwargs):

        super(Decoder, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        feat_channels = channels
        feat_strides = strides
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        num_queries = n_poly
        self.hidden_dim = em_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.pred_idx = levels_in - pred_layers - 1
        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        self.poly_dim = 2*(n_degree+1)
        
        # Transformer module
        n = sampling_points
        if atten_type == 'deform':
            n = num_points*sampling_points
        elif atten_type == 'poly+deform':
            n += num_points*sampling_points
        elif atten_type == 'polyv2':
            n = 0
        else:
            n = sampling_points
            print('Using Poly Attention')
            atten_type = 'poly'
        cross_atten = None
        self.s = s
        if atten_type == 'polyv2':
            cross_atten = PolyAttentionMechanismV2(em_dim, nhead, window_size=5, n_degree=n_degree)
        
        if use_poly_act:
            self.act = PolyAct(degree=n_degree, lower_bound=0.0, upper_bound=1.0)
            self.inv_act = InvPolyAct(degree=n_degree, lower_bound=0.0, upper_bound=1.0)
        else:
            self.act = nn.Identity()
            self.inv_act = nn.Identity()
        decoder_layer = TransformerDecoderLayer(em_dim, nhead, dim_feedforward, dropout, activation, num_levels, n, cross_atten=cross_atten, deform_levels=deform_levels)
        self.decoder = TransformerDecoder(em_dim, decoder_layer, num_decoder_layers, eval_idx=eval_idx, sampling_points=sampling_points, degree=n_degree, query_num=num_queries, num_points=num_points, atten_type=atten_type, act=self.act, inv_act=self.inv_act)

        self.n_degree = n_degree
        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.query_pos_head = MLP(self.poly_dim, 2 * self.hidden_dim, self.hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim,)
        )
        self.enc_score_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ELU(),
                                            nn.Linear(self.hidden_dim*2, 1),
                                            )
        # May Be change this adaptive Global pooling over the last level of the feature map and mlps to get he polys
        self.enc_bbox_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim*2), 
                                           nn.SiLU(),
                                           nn.Linear(self.hidden_dim*2, self.poly_dim),
                                         )
    
        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ELU(),
                                        nn.Linear(self.hidden_dim*2, 1),
                                        )
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.SiLU(),
                                        nn.Linear(self.hidden_dim*2, self.poly_dim),
                                        )
            for _ in range(num_decoder_layers)
        ])



        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        
        for layer in self.enc_score_head:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, bias)
        for layer in self.enc_bbox_head:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, bias)
        
        for layer in self.dec_score_head:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, bias)
        for layer in self.dec_bbox_head:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, bias)

        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        pass

    def _get_encoder_input(self, proj_feats):

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def sorted_search(self, levels, topk_idxs):
        levs = levels.unsqueeze(0).unsqueeze(1)
        topk_idxs = topk_idxs.unsqueeze(-1)
        return levels.shape[0] - (topk_idxs <= levs).sum(-1)
        
        


    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape

        output_memory = self.enc_output(memory)
        spatial_levels = torch.tensor([l[0]*l[1] for l in spatial_shapes], device=memory.device).cumsum(0)
        if(self.pred_idx < 0):
            start_idx = 0
        else:
            start_idx = spatial_levels[self.pred_idx]
        enc_outputs_class = self.enc_score_head(output_memory[..., start_idx:, :])
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory[..., start_idx:, :])

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        
        #levels = torch.searchsorted(spatial_levels, topk_ind) # This is find the level of the topk_ind
        levels = self.sorted_search(spatial_levels, topk_ind+start_idx)
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        
        enc_topk_polys = reference_points_unact
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])) # Best Logits

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        polys_detach = enc_topk_polys.detach()
        if denoising_bbox_unact is not None:
            B, DN, NP, DIM = denoising_bbox_unact.shape 
            polys_detach = torch.concat([denoising_bbox_unact.reshape(B, DN, -1), polys_detach], 1)
            levels = torch.concat([torch.randint(low=0, high=len(spatial_levels), size=(B, DN), device=levels.device).type(levels.type()),
                                   levels], 1)
        return target, polys_detach, self.act(enc_topk_polys), enc_topk_logits, levels

    def make_attn_mask_meta(self, denoising_groups, device):
        if(denoising_groups):
            all_queries = denoising_groups + [self.num_queries]
            group_attn = torch.kron(torch.eye(len(denoising_groups)), torch.ones(denoising_groups[0], denoising_groups[0]))
            querries_group_attn = torch.zeros((self.num_queries, group_attn.shape[-1]))
            group_attn = torch.cat([group_attn, querries_group_attn], dim=0)
            querries_querries_attn = torch.ones((sum(all_queries), self.num_queries))
            attn_mask = torch.cat([group_attn, querries_querries_attn], dim=1).to(device)
            dn_meta = {'all_queries': all_queries, 'num_split': [sum(denoising_groups), self.num_queries]}
            return (1.0-attn_mask).bool(), dn_meta
        return None, None 

    def forward(self, feats, targets=None, dn_queries=None, viz=False):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        B = memory.shape[0]
        
        if(not dn_queries):
            dn_queries = {
                'denoising_class' : None,
                'denoising_polys' : None,
                'denoising_groups' : None,
                'dn_meta' : None
            }

        denoising_class, denoising_bbox_unact = dn_queries['denoising_class'], dn_queries['denoising_polys']
        attn_mask, dn_meta = self.make_attn_mask_meta(dn_queries['denoising_groups'], feats[0].device)

        target, enc_topk_polys_detach, enc_topk_polys, enc_topk_logits, levels = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
    
        # decoder
        polys, logits, sampling_points, attn_weights = self.decoder(
            target,
            enc_topk_polys_detach,
            levels,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            viz=viz)
        out_polys = self.act(polys[-1, :, :, :])
        out_logits = logits[-1, :, :]
        # enc_topk_polys = self.act(enc_topk_polys)
        dn_out_polys = None
        dn_out_logits = None
        if self.training and dn_meta is not None:
            dn_out_polys, out_polys = torch.split(out_polys, dn_meta['num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['num_split'], dim=1)
            dn_out_polys  = dn_out_polys.view(-1, dn_meta['num_split'][0], 2, self.poly_dim//2).contiguous()
            dn_out_logits = dn_out_logits.view(-1, dn_meta['num_split'][0]).contiguous()
        if self.training:
            return out_polys.view(-1, self.num_queries, 2, self.poly_dim//2).contiguous(), out_logits.view(-1, self.num_queries).contiguous(), enc_topk_polys.view(-1, self.num_queries, 2, self.poly_dim//2).contiguous(), enc_topk_logits.view(-1, self.num_queries).contiguous(), dn_out_polys, dn_out_logits
        if viz:
            return out_polys.view(-1, self.num_queries, 2, self.poly_dim//2).contiguous(), out_logits.view(-1, self.num_queries).contiguous(), enc_topk_polys.view(-1, self.num_queries, 2, self.poly_dim//2).contiguous(), enc_topk_logits.view(-1, self.num_queries).contiguous(),\
                sampling_points, attn_weights
        return out_polys.view(B, self.num_queries, 2, self.poly_dim//2).contiguous(), out_logits.view(B, self.num_queries).contiguous()


