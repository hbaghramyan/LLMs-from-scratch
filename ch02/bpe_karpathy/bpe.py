# making the training text longer to have more representative token statistics
# text from https://www.reedbeta.com/blog/programmers-intro-to-unicode/
from utils import get_stats, merge, decode, encode

with open(r"ch02/bpe_karpathy/unicode.txt", "r", encoding="utf-8") as f:
    TEXT = f.read()

TOKENS = TEXT.encode("utf-8")
TOKENS = list(map(int, TOKENS))

# ---
vocab_size = 276  # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(TOKENS)  # copy so we don't destroy the original list

merges = {}  # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"mergin {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("tokens length:", len(TOKENS))
print("ids length:", len(ids))
print(f"compression ratio: {len(TOKENS) / len(ids):.2f}X")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

print(decode([128], vocab))

print(encode("Hello world!", merges))
