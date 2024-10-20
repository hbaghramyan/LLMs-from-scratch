import torch
import tiktoken
import os
import sys

sys.path.insert(0, os.getcwd())

from utils.utils_prev import GPTModel, generate_text_simple, create_dataloader_v1

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

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

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

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
loss_manual = -torch.Tensor(
    [
        torch.log(logit[target.item()]).item()
        for target, logit in zip(
            targets_flat, torch.nn.functional.softmax(logits_flat, dim=-1)
        )
    ]
).mean()

print(f"Torch version of {loss} vs manual version of it {loss_manual}")

# 5.1.3 Calculating the training and validation set losses

file_path = "ch05/01_main-chapter-code/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * total_characters)
train_data = text_data[:split_idx]
val_data = text_data[:split_idx]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    txt=train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    txt=val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

print("Train laoder:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# Saniti check

if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
    print(
        "Not enough tokens for the training loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "increase the `training_ratio`"
    )

if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
    print(
        "Not enough tokens for the validation loader. "
        "Try to lower the `GPT_CONFIG_124M['context_length']` or "
        "decrease the `training_ratio`"
    )

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        input=logits.flatten(0, 1), target=targets.flatten()
    )
    return loss
