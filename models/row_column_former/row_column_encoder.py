import torch
import torch.nn as nn
from models.row_column_former.utils import sinusoidal_positional_embedding
from models.utils import get_activation


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


    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        
        if self.normalize_before:
            src = self.norm1(src)
        q = k = sinusoidal_positional_embedding(src)
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

class RowColumnAttentionEncoder(nn.Module):
    def __init__(self, embed_dim, H, W, num_layers=3, num_heads=8):
        """
        Row-Column Attention Encoder.

        Args:
            embed_dim (int): Embedding dimension of the input features.
            num_layers (int): Number of encoder layers.
            num_heads (int): Number of attention heads in each layer.
        """
        super(RowColumnAttentionEncoder, self).__init__()
        
        # Transformer encoder layers for row and column attention
        self.row_attention_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.col_attention_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.row_proj = nn.Linear(embed_dim*H, embed_dim)
        self.col_proj = nn.Linear(embed_dim*W, embed_dim)
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, feature_map):
        """
        Forward pass for the RowColumnAttentionEncoder.

        Args:
            feature_map (Tensor): Input feature map of shape (B, C, H, W).
        
        Returns:
            Tensor: Encoded features of shape ((H + W), B, C).
        """
        B, C, H, W = feature_map.shape
        
        # Row-wise features: Average over width (columns)
        row_features = self.row_proj(feature_map.flatten(1, 2).permute(0, 2, 1).contiguous())  # (B, H, C)
        
        # Column-wise features: Average over height (rows)
        col_features = self.col_proj(feature_map.permute(0, 2, 1, 3).contiguous().flatten(2, 3))  # (B, W, C)
        
        # Add sinusoidal positional embeddings

        
        # Apply attention to rows and columns
        for row_layer, col_layer in zip(self.row_attention_layers, self.col_attention_layers):
            row_features = sinusoidal_positional_embedding(row_features)  # (H, C)
            col_features = sinusoidal_positional_embedding(col_features)  # (W, C)
            row_features = row_layer(row_features)  # (B, H, C)
            col_features = col_layer(col_features)  # (B, W, C)
        
        # Concatenate row and column features
        encoded_features = torch.cat([row_features, col_features], dim=1)  # (B, H + W, C)
        
        # Apply final layer normalization
        encoded_features = self.layer_norm(encoded_features)
        
        return encoded_features


if __name__ == "__main__":
    input_tensor = torch.randn(1, 128, 64, 32)
    model = RowColumnAttentionEncoder(embed_dim=128, H=64, W=32, num_layers=3, num_heads=8)
    output = model(input_tensor)
    print(output.shape)