import torch
import torch.nn as nn
from torch.nn import functional as F
from src.config import block_size, n_embd, dropout

#Self-Attention Head

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # Linear projections for key, query, and value
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower triangular matrix for masking (prevents looking at the future)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size (batch, time-step, channels)
        # output size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # 1. Compute attention scores ("affinities")
        # Fix: Scale by the square root of head_size (k.shape[-1]), not C
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)

        # 2. Apply causal mask (decoder-only transformer logic)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # 3. Normalize to probabilities
        wei = F.softmax(wei, dim=-1)

        # 4. Apply dropout to the attention weights
        wei = self.dropout(wei)

        # 5. Perform the weighted aggregation of the values
        v = self.value(x)   # (B, T, head_size)
        out = wei @ v       # (B, T, head_size)

        return out
    
    
# Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projects the concatenated heads back to n_embd
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # Standard dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate heads: (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply projection followed by dropout
        out = self.dropout(self.proj(out))
        return out
 
    
# Feed Forward Layer

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        dropout = 0.2

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
    
# Transformer Block

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x