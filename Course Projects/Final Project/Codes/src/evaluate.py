import json
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm

from config import PROCESSED_DIR, CHECKPOINT_DIR, METRICS_DIR, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_CHORD_TOKEN
from train import Vocab, MusicDataset, collate_fn
from model import Seq2SeqTransformer
from data_utils import event_tokens_to_notes, chord_to_pitch_classes

def note_set(notes):
    return {(n.onset, n.pitch, n.duration) for n in notes}

def onset_set(notes):
    return {(n.onset, n.pitch) for n in notes}

def prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

def chord_tone_ratio(notes, raw_chords):
    total = 0
    in_chord = 0
    for n in notes:
        pcs = []
        for c in raw_chords:
            if c["start"] <= n.onset < c["end"]:
                pcs = chord_to_pitch_classes(c["label"])
                break
        if len(pcs) == 0:
            continue
        total += 1
        if n.pitch % 12 in pcs:
            in_chord += 1
    return in_chord / (total + 1e-8)

def register_balance(notes):
    if len(notes) == 0:
        return {"lh_ratio": 0.0, "rh_ratio": 0.0}
    lh = sum(1 for n in notes if n.pitch < 60)
    rh = sum(1 for n in notes if n.pitch >= 60)
    total = len(notes)
    return {"lh_ratio": lh / total, "rh_ratio": rh / total}

@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_chords", action="store_true")
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    vocab = Vocab(PROCESSED_DIR / "vocab.json")
    test_ds = MusicDataset(PROCESSED_DIR / "test.jsonl", vocab, use_chords=args.use_chords)

    ckpt_name = "transformer_chords.pt" if args.use_chords else "transformer_melody.pt"
    ckpt = torch.load(CHECKPOINT_DIR / ckpt_name, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(
        vocab_size=len(vocab),
        d_model=256,
        nhead=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        pad_id=vocab.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_note_p, all_note_r, all_note_f1 = [], [], []
    all_on_p, all_on_r, all_on_f1 = [], [], []
    all_ctr = []
    all_lh_diff = []
    all_rh_diff = []

    for ex in tqdm(test_ds.examples, desc="Evaluating"):
        src = torch.tensor([ex["src_ids"]], dtype=torch.long, device=device)
        pred_ids = model.greedy_decode(src, vocab.bos_id, vocab.eos_id, max_len=1024)[0].tolist()

        # strip special tokens
        pred_toks = []
        for i in pred_ids[1:]:
            tok = vocab.itos.get(i, UNK_TOKEN)
            if tok == EOS_TOKEN:
                break
            if tok not in {PAD_TOKEN, BOS_TOKEN}:
                pred_toks.append(tok)

        gold_toks = vocab.decode(ex["tgt_ids"][1:-1])

        pred_notes = event_tokens_to_notes(pred_toks)
        gold_notes = event_tokens_to_notes(gold_toks)

        p, r, f1 = prf(note_set(pred_notes), note_set(gold_notes))
        op, or_, of1 = prf(onset_set(pred_notes), onset_set(gold_notes))

        all_note_p.append(p)
        all_note_r.append(r)
        all_note_f1.append(f1)
        all_on_p.append(op)
        all_on_r.append(or_)
        all_on_f1.append(of1)

        ctr = chord_tone_ratio(pred_notes, ex["raw_chords"])
        all_ctr.append(ctr)

        pred_reg = register_balance(pred_notes)
        gold_reg = register_balance(gold_notes)
        all_lh_diff.append(abs(pred_reg["lh_ratio"] - gold_reg["lh_ratio"]))
        all_rh_diff.append(abs(pred_reg["rh_ratio"] - gold_reg["rh_ratio"]))

        # raw chords are not stored in test_ds.examples; recover from JSONL directly
        # easiest robust way: use matching file from jsonl
        # here we add placeholders; the full code below loads from jsonl externally if needed
        # For now chord-tone ratio can be skipped if not available inside dataset class.
        # Better fix: attach raw_chords into MusicDataset.examples.

    metrics = {
        "note_precision": float(np.mean(all_note_p)),
        "note_recall": float(np.mean(all_note_r)),
        "note_f1": float(np.mean(all_note_f1)),
        "onset_precision": float(np.mean(all_on_p)),
        "onset_recall": float(np.mean(all_on_r)),
        "onset_f1": float(np.mean(all_on_f1)),
        "chord_tone_ratio": float(np.mean(all_ctr)),
        "lh_ratio_abs_error": float(np.mean(all_lh_diff)),
        "rh_ratio_abs_error": float(np.mean(all_rh_diff)),
    }

    out_name = "metrics_chords.json" if args.use_chords else "metrics_melody.json"
    out_path = METRICS_DIR / out_name
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[OK] saved metrics to {out_path}")

if __name__ == "__main__":
    main()