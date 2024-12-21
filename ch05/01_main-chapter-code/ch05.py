import torch
import tiktoken
import os
import sys
import urllib.request
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
# from previous_chapters import GPTModel, generate_text_simple, create_dataloader_v1
from utils.utils_prev import (
    GPTModel,
    generate,
    generate_text_simple,
    create_dataloader_v1,
)

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
    "cpu"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.eval()


# def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
#     """Convert text into tokens
#     Args:
#         text (str): input text
#         tokenizer (tiktoken.Encoding): encoding used to tokenize the input text
#     Returns:
#         encoded (torch.Tensor)
#     """
#     encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
#     encoded = torch.tensor(encoded).unsqueeze(0)
#     return encoded


# def token_ids_to_text(token_ids: torch.Tensor, tokenzer: tiktoken.Encoding) -> str:
#     """Convert input token ids into text
#     Args:
#         token_ids (torch.Tensor)
#         tokenizer (tiktoken.Encoding): encoding used to tokenize the input text
#     Returns:
#         decoded (str): decoded string of the input text + generated text
#     """
#     flat = token_ids.squeeze(0)
#     decoded = tokenzer.decode(flat.tolist())
#     return decoded


# start_context = "Every effort moves you"
# tokenizer = tiktoken.get_encoding("gpt2")

# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(start_context, tokenizer),
#     max_new_tokens=10,
#     context_size=GPT_CONFIG_124M["context_length"],
# )

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# # 5.1.2 Calculating the text generation loss

# # ["every effort moves"
# #  "I really like"]
# inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])

# # [" effort moves you",
# #  " really like chocolate"]
# targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

# with torch.no_grad():
#     logits = model(inputs)

# probas = torch.softmax(input=logits, dim=-1)

# token_ids = torch.argmax(input=probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)

# print("Targets batch 1:", token_ids_to_text(targets[0], tokenizer))
# print(
#     "Outputs batch 1:",
#     token_ids_to_text(token_ids[0].flatten(), tokenizer),
# )

# text_idx = 0
# target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 1:", target_probas_1)

# text_idx = 1
# target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 2:", target_probas_2)

# log_probas = torch.log(torch.cat(tensors=(target_probas_1, target_probas_2)))
# print(log_probas)

# avg_log_probas = torch.mean(log_probas)
# print(avg_log_probas)

# neg_avg_log_probas = avg_log_probas * -1

# print("Logits shape:", logits.shape)
# print("Targets shape:", targets.shape)

# logits_flat = logits.flatten(0, 1)
# targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)

# loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# loss_manual = -torch.Tensor(
#     [
#         torch.log(logit[target.item()]).item()
#         for target, logit in zip(
#             targets_flat, torch.nn.functional.softmax(logits_flat, dim=-1)
#         )
#     ]
# ).mean()

# print(f"Torch version of {loss} vs manual version of it {loss_manual}")

# # 5.1.3 Calculating the training and validation set losses

# file_path = "the-verdict.txt"
# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

# if not os.path.exists(file_path):
#     with urllib.request.urlopen(url) as response:
#         text_data = response.read().decode("utf-8")
#     with open(file_path, "w", encoding="utf-8") as file:
#         file.write(text_data)
# else:
#     with open(file_path, "r", encoding="utf-8") as file:
#         text_data = file.read()

# total_characters = len(text_data)
# total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)

# train_ratio = 0.90
# split_idx = int(train_ratio * total_characters)
# train_data = text_data[:split_idx]
# val_data = text_data[split_idx:]

# torch.manual_seed(123)
# train_loader = create_dataloader_v1(
#     txt=train_data,
#     batch_size=2,
#     max_length=GPT_CONFIG_124M["context_length"],
#     stride=GPT_CONFIG_124M["context_length"],
#     drop_last=True,
#     shuffle=True,
#     num_workers=0,
# )

# val_loader = create_dataloader_v1(
#     txt=val_data,
#     batch_size=2,
#     max_length=GPT_CONFIG_124M["context_length"],
#     stride=GPT_CONFIG_124M["context_length"],
#     shuffle=False,
#     drop_last=False,
#     num_workers=0,
# )

# print("Train laoder:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

# # Sanity check

# if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
#     print(
#         "Not enough tokens for the training loader. "
#         "Try to lower the `GPT_CONFIG_124M['context_length']` or "
#         "increase the `training_ratio`"
#     )

# if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
#     print(
#         "Not enough tokens for the validation loader. "
#         "Try to lower the `GPT_CONFIG_124M['context_length']` or "
#         "decrease the `training_ratio`"
#     )

# train_tokens = 0
# for input_batch, target_batch in train_loader:
#     train_tokens += input_batch.numel()

# val_tokens = 0
# for input_batch, target_batch in val_loader:
#     val_tokens += input_batch.numel()

# print("Training tokens:", train_tokens)
# print("Validation tokens:", val_tokens)
# print("All tokens:", train_tokens + val_tokens)


# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch = input_batch.to(device)
#     target_batch = target_batch.to(device)
#     logits = model(input_batch)
#     loss = torch.nn.functional.cross_entropy(
#         input=logits.flatten(0, 1), target=target_batch.flatten()
#     )
#     return loss


# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.0
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else:
#             break
#     return total_loss / num_batches


# model.to(device=device)
# torch.manual_seed(123)
# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device=device)
#     val_loss = calc_loss_loader(val_loader, model, device=device)
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)


# def train_model_simple(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     device,
#     num_epochs,
#     eval_freq,
#     eval_iter,
#     start_context,
#     tokenizer,
# ):
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen, global_step = 0, -1

#     for epoch in range(num_epochs):
#         model.train()
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             optimizer.step()
#             tokens_seen += input_batch.numel()
#             global_step += 1

#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter
#                 )
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 print(
#                     f"Ep {epoch+1} (Step {global_step:06d}):  "
#                     f"Train loss {train_loss:.3f}, "
#                     f"Val loss {val_loss:.3f}"
#                 )
#         generate_and_print_sample(model, tokenizer, device, start_context)
#     return train_losses, val_losses, track_tokens_seen


# def evaluate_model(model, train_loader, val_loader, device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(
#             train_loader, model, device, num_batches=eval_iter
#         )
#         val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
#     model.train()
#     return train_loss, val_loss


# def generate_and_print_sample(model, tokenizer, device, start_context):
#     model.eval()
#     context_size = model.pos_emb.weight.shape[0]
#     encoded = text_to_token_ids(start_context, tokenizer).to(device)
#     with torch.no_grad():
#         token_ids = generate_text_simple(
#             model=model, idx=encoded, max_new_tokens=50, context_size=context_size
#         )
#     decoded_text = token_ids_to_text(token_ids, tokenizer)
#     print(decoded_text.replace("\n", " "))
#     model.train()


# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     device,
#     num_epochs=num_epochs,
#     eval_freq=5,
#     eval_iter=5,
#     start_context="Every effort moves you",
#     tokenizer=tokenizer,
# )


# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

# # 5.3 Decoding strategies to control randomness

# model.to(device=device)
# model.eval()

# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text="Every effort moves you", tokenizer=tokenizer).to(
#         device
#     ),
#     max_new_tokens=25,
#     context_size=GPT_CONFIG_124M["context_length"],
# )

# print("Output text:\n", token_ids_to_text(token_ids=token_ids, tokenzer=tokenizer))

# 5.3.1 Temperature scaling

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
next_token = inverse_vocab[next_token_id]

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
next_token = inverse_vocab[next_token_id]
print(inverse_vocab[next_token_id])


def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


print_sampled_tokens(probas)


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
# x = torch.arange(len(vocab))
# bar_width = 0.15
# fig, ax = plt.subplots(figsize=(5, 3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(
#         x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}"
#     )

# ax.set_ylabel("Probability")
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()
# plt.tight_layout()
# plt.show()

top_k = 3
top_logits, top_pos = torch.topk(input=next_token_logits, k=top_k)
print("Top logits:", top_logits)
print("Top positionts", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits,
)

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)

print("Output text: \n", token_ids_to_text(token_ids, tokenizer))
