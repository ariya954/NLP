import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    PROCESSED_DIR, CHECKPOINT_DIR,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_CHORD_TOKEN,
    SRC_MAX_LEN, TGT_MAX_LEN, BATCH_SIZE, NUM_EPOCHS, LR,
    D_MODEL, NHEAD, NUM_LAYERS, FF_DIM, DROPOUT, PATIENCE,
    RANDOM_SEED
)
from model import Seq2SeqTransformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Vocab:
    def __init__(self, vocab_path: Path):
        obj = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.stoi = obj["stoi"]
        self.itos = {int(k): v for k, v in obj["itos"].items()}

        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        self.sep_chord_id = self.stoi[SEP_CHORD_TOKEN]

    def encode(self, toks):
        return [self.stoi.get(t, self.unk_id) for t in toks]

    def decode(self, ids):
        return [self.itos.get(int(i), UNK_TOKEN) for i in ids]

    def __len__(self):
        return len(self.stoi)

class MusicDataset(Dataset):
    def __init__(self, jsonl_path: Path, vocab: Vocab, use_chords: bool):
        self.examples = []
        self.vocab = vocab
        self.use_chords = use_chords

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)

                src_toks = ex["src_tokens"]
                if use_chords:
                    src_toks = src_toks + [SEP_CHORD_TOKEN] + ex["chord_tokens"]

                tgt_toks = ex["tgt_tokens"]

                src_ids = [vocab.bos_id] + vocab.encode(src_toks)[: SRC_MAX_LEN - 2] + [vocab.eos_id]
                tgt_ids = [vocab.bos_id] + vocab.encode(tgt_toks)[: TGT_MAX_LEN - 2] + [vocab.eos_id]

                self.examples.append({
                    "src_ids": src_ids,
                    "tgt_ids": tgt_ids,
                    "raw_chords": ex.get("raw_chords", []),
                    "meta": {
                        "song_id": ex["song_id"],
                        "segment_id": ex["segment_id"],
                    }
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, pad_id):
    max_src = max(len(x["src_ids"]) for x in batch)
    max_tgt = max(len(x["tgt_ids"]) for x in batch)

    src = []
    tgt = []
    meta = []

    for x in batch:
        src_ids = x["src_ids"] + [pad_id] * (max_src - len(x["src_ids"]))
        tgt_ids = x["tgt_ids"] + [pad_id] * (max_tgt - len(x["tgt_ids"]))
        src.append(src_ids)
        tgt.append(tgt_ids)
        meta.append(x["meta"])

    return {
        "src": torch.tensor(src, dtype=torch.long),
        "tgt": torch.tensor(tgt, dtype=torch.long),
        "meta": meta,
    }

def loss_fn(logits, tgt_out, pad_id):
    vocab_size = logits.size(-1)
    return nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        tgt_out.reshape(-1),
        ignore_index=pad_id
    )

def run_epoch(model, loader, optimizer, device, pad_id, train=True):
    model.train(train)
    total_loss = 0.0
    count = 0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.set_grad_enabled(train):
            logits = model(src, tgt_in)
            loss = loss_fn(logits, tgt_out, pad_id)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        count += 1
        pbar.set_description(f"{'train' if train else 'valid'} loss={loss.item():.4f}")

    return total_loss / max(count, 1)

def train_model(use_chords: bool):
    set_seed(RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = Vocab(PROCESSED_DIR / "vocab.json")
    train_ds = MusicDataset(PROCESSED_DIR / "train.jsonl", vocab, use_chords=use_chords)
    valid_ds = MusicDataset(PROCESSED_DIR / "valid.jsonl", vocab, use_chords=use_chords)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id)
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        pad_id=vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_epoch = -1
    patience_count = 0
    ckpt_name = "transformer_chords.pt" if use_chords else "transformer_melody.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, device, vocab.pad_id, train=True)
        va_loss = run_epoch(model, valid_loader, optimizer, device, vocab.pad_id, train=False)

        print(f"[Epoch {epoch}] train={tr_loss:.4f} valid={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab.stoi,
                "use_chords": use_chords,
            }, ckpt_path)
            print(f"[OK] saved best checkpoint to {ckpt_path}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[STOP] early stopping at epoch {epoch}")
                break

    print(f"[DONE] best epoch={best_epoch}, best valid={best_val:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_chords", action="store_true")
    args = parser.parse_args()
    train_model(use_chords=args.use_chords)