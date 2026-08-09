import json
from collections import Counter
from pathlib import Path

from config import (
    PROCESSED_DIR, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_CHORD_TOKEN
)

def main():
    train_path = PROCESSED_DIR / "train.jsonl"
    vocab_path = PROCESSED_DIR / "vocab.json"

    counter = Counter()
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            counter.update(ex["src_tokens"])
            counter.update(ex["tgt_tokens"])
            counter.update(ex["chord_tokens"])

    specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_CHORD_TOKEN]
    stoi = {tok: i for i, tok in enumerate(specials)}
    for tok, _ in counter.most_common():
        if tok not in stoi:
            stoi[tok] = len(stoi)

    itos = {i: tok for tok, i in stoi.items()}

    vocab = {"stoi": stoi, "itos": itos}
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"[OK] vocab size = {len(stoi)}")
    print(f"[OK] saved to {vocab_path}")

if __name__ == "__main__":
    main()