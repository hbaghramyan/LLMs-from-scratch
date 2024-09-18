import torch
from thop import profile
import os
import psutil
import sys

sys.path.insert(0, os.getcwd())

from utils.utils_prev import GPTModel

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (335M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS Memory: {mem_info.rss / 1024**2:.2f} MB")
    print(f"VMS Memory: {mem_info.vms / 1024**2:.2f} MB")
    print("-" * 30)


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
input_tensor = torch.randint(0, 50257, (2, 1024)).to(device)

for size in model_configs:
    BASE_CONFIG.update(model_configs[size])

    print("Profiling memory usage", size)

    model = GPTModel(BASE_CONFIG)
    model.to(device)

    print_memory_usage()
    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, _ = profile(model=model, inputs=(input_tensor,), verbose=False)
    flops = 2 * macs

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    print_memory_usage()
