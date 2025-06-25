import torch
import math

def sinusoidal_positional_embedding(input_tensor):
    """
    Generates sinusoidal positional embeddings.

    Args:
        input_tensor (Tensor): Input tensor of shape (B, L, C), where:
                               B: Batch size,
                               L: Sequence length (rows or columns),
                               C: Embedding dimension (feature size).
    
    Returns:
        Tensor: Positional embeddings of shape (L, C), broadcastable across the batch dimension.
    """
    # Extract sequence length (L) and embedding dimension (C) from input tensor
    _, L, C = input_tensor.shape
    
    # Compute position indices
    position = torch.arange(0, L, dtype=torch.float).unsqueeze(1).to(input_tensor.device)  # (L, 1)
    
    # Compute scaling factors
    div_term = torch.exp(torch.arange(0, C, 2, dtype=torch.float).to(input_tensor.device) *
                         (-math.log(10000.0) / C))  # (C // 2)
    
    # Initialize positional embedding
    pos_embedding = torch.zeros((L, C), device=input_tensor.device)  # (L, C)
    
    # Apply sine to even indices and cosine to odd indices
    pos_embedding[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
    pos_embedding[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
    
    return pos_embedding.unsqueeze(0) + input_tensor 


if __name__ == "__main__":
    input_tensor = torch.randn(1, 10, 10)
    output = sinusoidal_positional_embedding(input_tensor)
    print(output.shape)