import torch
import tiktoken
import os
import sys

sys.path.insert(0, os.getcwd())

from utils.utils_prev import GPTModel, generate_text_simple


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
    flat = token_ids.squeeze(0)
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
