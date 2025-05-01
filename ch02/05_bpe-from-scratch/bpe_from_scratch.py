import os
import urllib.request

import tiktoken

from utils import BPETokenizerSimple


def main():
    path = r"ch02/01_main-chapter-code/the-verdict.txt"

    gpt2_tokenizer = tiktoken.get_encoding("gpt2")

    for i in range(300):
        decoded = gpt2_tokenizer.decode([i])
        print(f"{i}: {decoded}")

    for i in range(300):
        decoded = gpt2_tokenizer.decode([i])
        print(f"{i}: {decoded}")

    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        file_path = path
        urllib.request.urlretrieve(url, file_path)

    with open(path, "r", encoding="utf-8") as f:  # added ../01_main-chapter-code/
        text = f.read()

    tokenizer = BPETokenizerSimple()
    tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})

    # print(tokenizer.vocab)
    print(len(tokenizer.vocab))

    print(len(tokenizer.bpe_merges))

    input_text = "Jack embraced beauty through art and life.\nMarie loves Jack"
    token_ids = tokenizer.encode(input_text)
    print(token_ids)

    tokenizer.decode(token_ids)


if __name__ == "__main__":
    main()
