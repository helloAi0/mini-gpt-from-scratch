from src.model import GPTLanguageModel
from src.dataset import get_batch, vocabulary_size
from src.config import *
import torch


model = GPTLanguageModel(vocabulary_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step}, loss {loss.item():.4f}")