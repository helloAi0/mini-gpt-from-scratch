
from model import GPTLanguageModel
from dataset import decode
import torch

model = GPTLanguageModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

context = torch.zeros((1,1), dtype=torch.long)
generated = model.generate(context, 200)
print(decode(generated[0].tolist()))