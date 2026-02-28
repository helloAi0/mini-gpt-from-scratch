import torch

device = 'cpu'

block_size = 128   #256
batch_size = 32
n_embd = 128
n_layer = 4
n_head = 4
dropout = 0.1

learning_rate = 3e-4
max_steps = 30000
eval_interval = 1500
weight_decay = 0.01