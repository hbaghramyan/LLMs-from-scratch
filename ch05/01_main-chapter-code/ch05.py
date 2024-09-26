import torch
import tiktoken
import os
import sys

sys.path.insert(0, os.getcwd())

from utils.utils_prev import GPTModel, generate_text_simple

# 5.1.1 Using GPT to generate text

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Convert text into tokens
    Args:
        text (str): input text
        tokenizer (tiktoken.Encoding): encoding used to tokenize the input text
    Returns:
        encoded (torch.Tensor)
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded = torch.tensor(encoded).unsqueeze(0)
    return encoded


def token_ids_to_text(token_ids: torch.Tensor, tokenzer: tiktoken.Encoding) -> str:
    """Convert input token ids into text
    Args:
        token_ids (torch.Tensor)
        tokenizer (tiktoken.Encoding): encoding used to tokenize the input text
    Returns:
        decoded (str): decoded string of the input text + generated text
    """
    flat = token_ids.squeeze()
    decoded = tokenzer.decode(flat.tolist())
    return decoded


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer=tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids=token_ids, tokenzer=tokenizer))

# 5.1.2 Calculating the text generation loss

# ["every effort moves"
#  "I really like"]
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])

# [" effort moves you",
#  " really like chocolate"]
targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(input=logits, dim=-1)

token_ids = torch.argmax(input=probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

print("Targets batch 1:", token_ids_to_text(token_ids=targets[0], tokenzer=tokenizer))
print(
    "Outputs batch 1:",
    token_ids_to_text(token_ids=token_ids[0], tokenzer=tokenizer),
)

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat(tensors=(target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
