# Tokenizer loading
# Streaming dataset

from transformers import AutoTokenizer
from datasets import load_dataset
from src.config import batch_size, block_size, device

# 1. Load the dataset in streaming mode
ds = load_dataset("Skylion007/openwebtext", split='train', streaming=True)

#BPE via HuggingFace
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

vocabulary_size = tokenizer.vocab_size
print("Vocab size:", vocabulary_size)

def encode(text):
    return tokenizer.encode(text)

def decode(tokens):
    return tokenizer.decode(tokens)

# get_batch function
# encode/decode

import torch

# Create iterators for the stream
# We use a shuffle buffer so the model doesn't see documents in a fixed order
train_stream = ds.shuffle(seed=42, buffer_size=1000)
train_iter = iter(train_stream)

def get_batch(split):
    global train_iter # Moved global declaration to the start of the function
    # In a stream, we just pull the next documents
    # For a simple 'val' split on a stream, we could just take every 10th document
    x_list, y_list = [], []

    while len(x_list) < batch_size:
        try:
            example = next(train_iter)
            text = example['text']

            if len(text) > block_size + 1:
                # Encode the text on the fly
                encoded_text = encode(text)
                if len(encoded_text) < block_size + 1:
                  continue
                data = torch.tensor(encoded_text, dtype=torch.long)

                # Pick a random starting point in this document
                # Fix: The upper bound for randint should be len(data) - block_size
                # to allow `i` to be 0 when len(data) == block_size + 1
                if len(data) - block_size <= 0:
                    # Skip this data point if it's too short after encoding
                    continue
                i = torch.randint(0, len(data) - block_size, (1,)).item()

                x_list.append(data[i : i+block_size])
                y_list.append(data[i+1 : i+block_size+1])
        except StopIteration:
            # Restart the stream if it ends
            train_iter = iter(train_stream)

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x.to(device), y.to(device)


# # Pull a single batch to inspect the data
# # x, y = get_batch('train')

# print('inputs shape:', x.shape) # Should be (batch_size, block_size)
# print('inputs:')
# print(x)

# print('targets shape:', y.shape) # Should be (batch_size, block_size)
# print('targets:')
# print(y)

# # To see what the model actually "sees" for the first sequence in the batch:
# print("\nFirst sequence decoded:")
# print(decode(x[0].tolist()))