import json
from pathlib import Path
from train import Vocab, MusicDataset
from config import PROCESSED_DIR

vocab_path = PROCESSED_DIR / "vocab.json"
data_path = PROCESSED_DIR / "valid.jsonl" # or "train.jsonl" or "test.jsonl"

vocab = Vocab(vocab_path)
dataset = MusicDataset(data_path, vocab, use_chords=True)

# Load and inspect the first three "songs" (segments)
for i in range(3):
    example = dataset[i]
    
    src_tokens = vocab.decode(example["src_ids"])
    tgt_tokens = vocab.decode(example["tgt_ids"])
    
    print(f"--- Song Segment {i+1} (ID: {example['meta']['song_id']}) ---")
    
    # Print lengths (Note: these include BOS and EOS tokens added by the Dataset)
    print(f"Number of melody (src) tokens: {len(src_tokens)}")
    print(f"Number of piano (tgt) tokens:  {len(tgt_tokens)}")
    
    # Print first 20 tokens
    print(f"First 20 src tokens: {src_tokens[:20]}")
    print(f"First 20 tgt tokens: {tgt_tokens[:20]}")
    print("\n")