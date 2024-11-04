import torch
import torch.nn as nn
import torch.nn.functional as F

# class ResidualConnection(nn.Module):

#     def __init__(self, features: int, dropout: float) -> None:
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.norm = LayerNormalization(features)

#     def forward(self, x, sublayer):
#         return x + self.dropout(sublayer(self.norm(x)))

# class LayerNormalization(nn.Module):

#     def __init__(self, features: int, eps: float = 10**-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(
#             torch.ones(features)
#         )  # alpha is a learnable parameter
#         self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

#     def forward(self, x):
#         # x: (batch, seq_len, hidden_size)
#         # Keep the dimension for broadcasting
#         mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
#         # Keep the dimension for broadcasting
#         std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
#         # eps is to prevent dividing by zero or when std is very small
#         return self.alpha * (x - mean) / (std + self.eps) + self.bias

# class FeedForwardBlock(nn.Module):

#     def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
#         super().__init__()
#         self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

#     def forward(self, x):
#         # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
#         return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_heads=8):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Linear layers to transform the input dimension to hidden_dim and back
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
        # Attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, query, key, value, mask, causal_mask):
        # Transform the input dimensions to hidden_dim
        query = self.fc1(query)
        key = self.fc1(key)
        value = self.fc1(value)
        
        # Apply attention
        query = self.self_attention(query, query, query, key_padding_mask=mask[:, :query.size(1)], attn_mask=causal_mask.bool().repeat(self.num_heads, 1, 1))[0]
        attn_output = self.cross_attention(query, key, value, key_padding_mask=mask)[0]
        
        output = self.fc2(attn_output)
        
        return output

# Example usage
if __name__ == "__main__":
    batch_size = 16
    seq_len = 10
    input_dim = 256
    
    # Create random input tensors of shape (batch_size, seq_len, input_dim)
    query = torch.randn(batch_size, seq_len, input_dim)
    key = torch.randn(batch_size, seq_len, input_dim)
    value = torch.randn(batch_size, seq_len, input_dim)
    
    # Initialize the attention module
    attention_module = Attention(input_dim=input_dim)
    
    # Apply the attention module
    output = attention_module(query, key, value)
    
    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)
    print("Output shape:", output.shape)