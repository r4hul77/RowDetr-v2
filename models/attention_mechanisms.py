import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyAttentionMechanism(nn.Module):
    def __init__(self, feature_dim, num_heads, window_size, n_degree):
        super(PolyAttentionMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (feature_dim // num_heads) ** -0.5
        self.n_degree = n_degree
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.conv1x1 = nn.Conv2d(self.window_size * self.window_size, 1, kernel_size=1)
        # Assuming a simple learned positional bias
        self.positional_bias = nn.Parameter(torch.randn(window_size * window_size, 1))
        self.mda = MultiDimensionalAttention(num_points=self.n_degree+1, channel_dim=feature_dim, n_degree=n_degree)
        alpha = torch.linspace(0, 1, self.n_degree+1)
        self.alphas = torch.stack([alpha ** i for i in range(self.n_degree, -1, -1)], 1)
    
    def get_points(self, polys):
        return (polys@(self.alphas.t().to(polys.device).type(polys.type()))).transpose(-1, -2).contiguous()
    
    def forward(self, x, polys):
        points = self.get_points(polys)
        B, C, H, W = x.shape
        _, Q, Np, _ = points.shape
        
        # Scale points from [0, 1] to [-1, 1] for grid_sample compatibility
        points = 2 * points - 1  # Now in range [-1, 1]
        points = points.unsqueeze(dim=2)  # [B, Q, 1, Np, 2]

        # Create a grid around each point
        grid_size = self.window_size // 2
        delta_h = 2*torch.linspace(-self.window_size // 2, self.window_size // 2, self.window_size).to(x.device) / H
        delta_w = 2*torch.linspace(-self.window_size // 2, self.window_size // 2, self.window_size).to(x.device) / W
        mesh_x, mesh_y = torch.meshgrid(delta_w, delta_h, indexing="xy")
        mesh_x = mesh_x.reshape(-1, 1)
        mesh_y = mesh_y.reshape(-1, 1)
        
        grid_y = points[..., 0] + mesh_y
        grid_x = points[..., 1] + mesh_x
        
        grid = torch.stack((grid_y, grid_x), dim=-1)
        grid = grid.view(B * Q * Np, self.window_size, self.window_size, 2)
        
        


        # Reshape x for grid sampling
        x_grid = x.view(B, C, H, W).repeat(1, Q * Np, 1, 1).view(B * Q * Np, C, H, W)
        sampled_patches = F.grid_sample(x_grid, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Reshape to extract features for attention
        sampled_patches = sampled_patches.view(B, Q, Np, C, self.window_size * self.window_size).permute(0, 1, 2, 4, 3).reshape(B, Q * Np, self.window_size * self.window_size, C)

        # QKV
        qkv = self.qkv(sampled_patches).reshape(B, Q * Np, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # [3, B, num_heads, Q * Np, num_features, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.positional_bias.view(1, 1, 1, -1, 1)
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        crcts = (attn @ v).transpose(2, 3).reshape(B, Q * Np, -1, C).transpose(1, 2)
        crcts = self.proj(self.conv1x1(crcts).squeeze(dim=1))

        return crcts.reshape(B, Q, Np, -1)  # Reshape to separate queries and points
class MultiDimensionalAttention(nn.Module):
    def __init__(self, num_points, channel_dim, n_degree):
        super(MultiDimensionalAttention, self).__init__()
        self.num_points = num_points
        self.channel_dim = channel_dim
        self.n_degree = n_degree

        # Linear layers to transform inputs for the multi-dimensional attention
        self.query_transform = nn.Linear(channel_dim, channel_dim)
        self.key_transform = nn.Linear(channel_dim, channel_dim)
        self.value_transform = nn.Linear(channel_dim, channel_dim)

        # Multi-dimensional attention combines spatial and channel dimensions
        self.attention = nn.MultiheadAttention(embed_dim=channel_dim, num_heads=8)

        # Final mapping to polynomial coefficients
        self.coefficients = nn.Linear(num_points * channel_dim, 2 * (n_degree + 1))

    def forward(self, x):
        B, Q, Np, C = x.shape
        x = x.view(B * Q, Np, C)  # Combine batch and query dimensions for parallel processing

        # Apply transformations
        queries = self.query_transform(x)
        keys = self.key_transform(x)
        values = self.value_transform(x)

        # Apply attention
        attn_output, _ = self.attention(queries, keys, values)  # output shape: [B*Q, Np, C]
        attn_output = attn_output.view(B, Q, Np * C)  # Flatten spatial and channel dimensions

        # Map to polynomial coefficients
        coefficients = self.coefficients(attn_output).view(B, Q, 2, self.n_degree + 1)
        
        return coefficients

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyAttentionMechanismV2(nn.Module):
    def __init__(self, feature_dim, num_heads, window_size, n_degree):
        super(PolyAttentionMechanismV2, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (feature_dim // num_heads) ** -0.5

        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.conv1x1 = nn.Conv2d(self.window_size * self.window_size, 1, kernel_size=1)
        self.n_degree = n_degree
        # Assuming a simple learned positional bias
        self.positional_bias = nn.Parameter(torch.randn(window_size * window_size, 1))
        self.convkk = nn.Conv2d(feature_dim*(self.n_degree+1), feature_dim, kernel_size=self.window_size, padding=0)
        alphas = torch.linspace(0, 1, self.n_degree+1)
        self.alphas = torch.stack([alphas ** i for i in range(n_degree, -1, -1)], 1)
    def get_points(self, polys):
        B, Q, _, ND = polys.shape # polys are in the shape of [B, Q, (x, y), ND]
        all_pnts = []
        points = polys.unsqueeze(dim=-2)@self.alphas.t().to(polys.device).type(polys.type())
        return points


    def forward(self, query, polys, memory, memory_spatial_shapes, memory_mask=None):
        B, LEN_V, C = memory.shape
        x = memory[:, :memory_spatial_shapes[0][0]*memory_spatial_shapes[0][1], :].transpose(-2, -1).reshape(B, C, memory_spatial_shapes[0][0], memory_spatial_shapes[0][1]).contiguous()
        
        B, C, H, W = x.shape
        _, Q, _ = polys.shape
        points = self.get_points(polys.view(B, Q, 2, self.n_degree+1)).squeeze(dim=-2).transpose(-1, -2).flip(dims=(-1,))
        B, Q, Np, _ = points.shape # W, H format
        
        
        points = 2 * points - 1  # Now in range [-1, 1]
        points = points.unsqueeze(dim=2)  # [B, Q, 1, Np, 2]

        # Create a grid around each point
        grid_size = self.window_size // 2
        delta_h = 2*torch.linspace(-self.window_size // 2, self.window_size // 2, self.window_size).to(x.device) / H
        delta_w = 2*torch.linspace(-self.window_size // 2, self.window_size // 2, self.window_size).to(x.device) / W
        mesh_x, mesh_y = torch.meshgrid(delta_w, delta_h, indexing="xy")
        mesh_x = mesh_x.reshape(-1, 1)
        mesh_y = mesh_y.reshape(-1, 1)
        
        grid_x = points[..., 1] + mesh_x
        grid_y = points[..., 0] + mesh_y
        
        grid = torch.stack((grid_y, grid_x), dim=-1)
        grid = grid.view(B,  Q * Np *self.window_size, self.window_size, 2)
        
    


        # Reshape x for grid sampling
        sampled_patches = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True) # B, C, Q*Np* H, W

        # Reshape to extract features for attention
        sampled_patches = sampled_patches.view(B, C, Q*Np, self.window_size, self.window_size).contiguous() # B, C, Q*Np, H, W
        
        sampled_patches = sampled_patches.permute(0, 2, 1, 3, 4).contiguous() # B, Q*Np, C, H, W
        
        sampled_patches = sampled_patches.reshape(B*Q, Np*C, self.window_size, self.window_size).contiguous()# B, Q*Np, C, H*W

        crcts = F.relu(self.convkk(sampled_patches)).squeeze(dim=(-2)).squeeze(dim=(-1)) # B, 1, Q, Np, C, H, W

        # QKV

        crcts = self.proj(crcts).view(B, Q, C)

        return crcts, points 