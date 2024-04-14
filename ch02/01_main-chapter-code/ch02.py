# standard library inputs
import re

# third-partt library inputs
import tiktoken
import torch

# local module imports
from utils import create_dataloader

with open(r"ch02/01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'(?:[,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item]

tokenizer_bpt = tiktoken.get_encoding("gpt2")

output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=5, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

token_embeddings = token_embedding_layer(inputs)

block_size = max_length
pos_embedding_layer = torch.nn.Embedding(block_size, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(block_size))

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)