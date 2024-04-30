# standard library inputs
import re

# third-partt library inputs
import tiktoken
import torch

# local module imports
from utils import SimpleTokenizerV2, create_dataloader

with open(r"ch02/01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'(?:[,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item]

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

tokenizer_bpt = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
)
integers = tokenizer_bpt.encode(text, allowed_special={"<|endoftext|>"})
string = tokenizer_bpt.decode(integers)
print(string)

# 2.6 Data sampling with a sliding window

with open(r"ch02/01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer_bpt.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

x = enc_text[:context_size]
y = enc_text[1:context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer_bpt.decode(context), "---->", tokenizer_bpt.decode([desired]))

input_ids = torch.tensor([5, 1, 4, 3])

output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.EmbeddingBag(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader(
    raw_text, batch_size=8, max_length=max_length, stride=5, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
