import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear layers to transform the input dimension to hidden_dim and back
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, query, key, value, mask):
        # Transform the input dimensions to hidden_dim
        query = self.fc1(query)
        key = self.fc1(key)
        value = self.fc1(value)
        
        # Apply attention
        attn_output, _ = self.attention(query, key, value, key_padding_mask=mask)
        
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