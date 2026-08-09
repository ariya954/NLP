import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from config import (
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    METRICS_DIR,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    SEP_CHORD_TOKEN,
    D_MODEL,
    NHEAD,
    NUM_LAYERS,
    FF_DIM,
    DROPOUT,
    TGT_MAX_LEN,
)
from train import Vocab, MusicDataset
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

def get_active_chord_pcs(onset, raw_chords):
    for c in raw_chords:
        if c["start"] <= onset < c["end"]:
            return chord_to_pitch_classes(c["label"])
    return []

def clean_notes_basic(notes):
    """
    Minimal safe cleanup:
    1) drop invalid durations
    2) clamp pitch into piano range
    3) merge duplicates with same (onset, pitch), keep longest duration
    """
    cleaned = []
    for n in notes:
        if getattr(n, "duration", 0) <= 0:
            continue

        # clamp note properties
        n.onset = int(n.onset)
        n.pitch = max(21, min(108, int(n.pitch)))
        n.duration = int(n.duration)

        cleaned.append(n)

    best = {}
    for n in cleaned:
        key = (n.onset, n.pitch)
        if key not in best:
            best[key] = n
        else:
            if n.duration > best[key].duration:
                best[key] = n

    out = list(best.values())
    out.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    return out

def clean_notes_chord_snap(notes, raw_chords, max_snap=1):
    """
    Conservative harmonic cleanup:
    If a note is off-chord and within +-max_snap semitones of a chord tone,
    move it to the nearest chord tone.
    """
    for n in notes:
        pcs = get_active_chord_pcs(n.onset, raw_chords)
        if not pcs:
            continue

        if n.pitch % 12 in pcs:
            continue

        best_pitch = n.pitch
        best_dist = 999

        for delta in range(-max_snap, max_snap + 1):
            cand = n.pitch + delta
            if cand < 21 or cand > 108:
                continue
            if cand % 12 in pcs:
                if abs(delta) < best_dist:
                    best_dist = abs(delta)
                    best_pitch = cand

        if best_dist <= max_snap:
            n.pitch = best_pitch

    return clean_notes_basic(notes)

@torch.no_grad()
def beam_search_decode(model, src, bos_id, eos_id, max_len=1024, beam_size=3, device="cpu"):
    """
    Minimal beam search for batch size 1.
    Assumes model(src, tgt) returns logits of shape [B, T, V].
    """
    beams = [(torch.tensor([[bos_id]], dtype=torch.long, device=device), 0.0, False)]

    for _ in range(max_len - 1):
        candidates = []
        all_finished = True

        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, True))
                continue

            all_finished = False

            logits = model(src, seq)[:, -1, :]  # [1, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            for k in range(beam_size):
                tok_id = topk_ids[0, k].item()
                tok_lp = topk_log_probs[0, k].item()

                new_seq = torch.cat(
                    [seq, torch.tensor([[tok_id]], dtype=torch.long, device=device)],
                    dim=1,
                )
                new_finished = tok_id == eos_id
                candidates.append((new_seq, score + tok_lp, new_finished))

        if all_finished:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

        if all(finished for _, _, finished in beams):
            break

    best_seq = beams[0][0]
    return best_seq[0].tolist()

@torch.no_grad()
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_chords", action="store_true")
    parser.add_argument("--beam_size", type=int, default=1, help="1 = greedy, >1 = beam search")
    parser.add_argument("--max_examples", type=int, help="evaluate only first N examples")
    parser.add_argument("--clean_basic", action="store_true", help="apply basic output cleaning")
    parser.add_argument("--clean_chord_snap", action="store_true", help="apply conservative chord snapping")
    parser.add_argument("--snap_semitones", type=int, default=1, help="max semitone snap for chord cleanup")
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    vocab = Vocab(PROCESSED_DIR / "vocab.json")
    test_ds = MusicDataset(PROCESSED_DIR / "test.jsonl", vocab, use_chords=args.use_chords)
    examples = test_ds.examples[: args.max_examples]

    ckpt_name = "transformer_chords.pt" if args.use_chords else "transformer_melody.pt"
    ckpt = torch.load(CHECKPOINT_DIR / ckpt_name, map_location="cpu")

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

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_note_p, all_note_r, all_note_f1 = [], [], []
    all_on_p, all_on_r, all_on_f1 = [], [], []
    all_ctr = []
    all_lh_diff = []
    all_rh_diff = []

    desc = f"Evaluating ({len(examples)} ex, beam={args.beam_size}"
    if args.clean_basic:
        desc += ", basic_clean"
    if args.clean_chord_snap:
        desc += f", snap={args.snap_semitones}"
    desc += ")"

    for ex in tqdm(examples, desc=desc):
        src = torch.tensor([ex["src_ids"]], dtype=torch.long, device=device)

        if args.beam_size == 1:
            pred_ids = model.greedy_decode(
                src,
                vocab.bos_id,
                vocab.eos_id,
                max_len=TGT_MAX_LEN,
            )[0].tolist()
        else:
            pred_ids = beam_search_decode(
                model,
                src,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_len=TGT_MAX_LEN,
                beam_size=args.beam_size,
                device=device,
            )

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

        # output cleanup
        if args.clean_basic:
            pred_notes = clean_notes_basic(pred_notes)

        if args.clean_chord_snap:
            # chord snap already applies basic cleanup again
            pred_notes = clean_notes_chord_snap(
                pred_notes,
                ex["raw_chords"],
                max_snap=args.snap_semitones,
            )

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

    metrics = {
        "num_examples": len(examples),
        "beam_size": args.beam_size,
        "use_chords": args.use_chords,
        "clean_basic": args.clean_basic,
        "clean_chord_snap": args.clean_chord_snap,
        "snap_semitones": args.snap_semitones if args.clean_chord_snap else None,
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

    mode = "chords" if args.use_chords else "melody"
    clean_tag = "raw"
    if args.clean_basic and not args.clean_chord_snap:
        clean_tag = "basicclean"
    elif args.clean_chord_snap:
        clean_tag = f"chordsnap{args.snap_semitones}"

    out_name = f"metrics_{mode}_beam{args.beam_size}_{len(examples)}ex_{clean_tag}.json"
    out_path = METRICS_DIR / out_name
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"[OK] saved metrics to {out_path}")

if __name__ == "__main__":
    main()