import torch
import torch.nn as nn
from torch.nn import functional as F
from src.attention import Block
from src.config import n_embd, n_layer, n_head, block_size

# GPTLanguageModel

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token looks up a vector for its identity
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Each position (0 to block_size) gets its own learned vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Multiple layers of Transformer Blocks (Communication + Computation)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Final layer normalization to stabilize the output
        self.ln_f = nn.LayerNorm(n_embd)
        # Maps hidden states back to vocabulary scores (logits)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Better weight initialization for deep networks
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # tok_emb: (B, T, n_embd), pos_emb: (T, n_embd)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        # Combine identity and position info; process through blocks
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Flatten 3D to 2D for cross_entropy compatibility
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop current context to fit within model's memory (block_size)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Focus only on the very last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution to pick the next character
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append predicted index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# #Initialize model:
# model = GPTLanguageModel(vocabulary_size)
# model = model.to(device)