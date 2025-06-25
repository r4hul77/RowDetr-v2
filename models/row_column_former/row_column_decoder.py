import torch
import torch.nn as nn
from .row_column_encoder import TransformerEncoderLayer
from models.utils import get_activation

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.normalize_before = normalize_before

        # Override the initialization from parent class
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                ):
        tgt2 = self.norm1(tgt) if self.normalize_before else tgt
        q = k = self.with_pos_embed(tgt2)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt) if self.normalize_before else tgt

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2),
                                   key=self.with_pos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt) if self.normalize_before else tgt

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor):
        return sinusoidal_positional_embedding(tensor)
        
        


from models.row_column_former.utils import sinusoidal_positional_embedding
class RowColumnAttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_queries=6, num_layers=3, num_heads=8):
        """
        Row-Column Attention Decoder.

        Args:
            embed_dim (int): Dimension of embeddings for the input features.
            num_queries (int): Number of learnable object queries.
            num_layers (int): Number of decoder layers.
            num_heads (int): Number of attention heads in each layer.
        """
        super(RowColumnAttentionDecoder, self).__init__()
        
        # Learnable object queries
        self.object_queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        self.enc_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, memory):
        """
        Forward pass for the RowColumnAttentionDecoder.

        Args:
            memory (Tensor): Encoded features from the encoder of shape (B, S, C), where:
                             B: Batch size,
                             S: Sequence length (H + W from encoder),
                             C: Embedding dimension.
        
        Returns:
            Tensor: Decoded features of shape (B, num_queries, embed_dim).
        """
        B, _, C = memory.size()
        
        # Expand object queries to match the batch size
        queries = self.object_queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, num_queries, embed_dim)
        queries = self.enc_layer(sinusoidal_positional_embedding(queries))
        # Apply transformer decoder layers
        for layer in self.layers:
            queries = layer(tgt=queries, memory=memory)
        
        # Apply final layer normalization
        decoded_features = self.layer_norm(queries)
        
        return decoded_features
