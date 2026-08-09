import json
from pathlib import Path

import torch
import pretty_midi

from config import PROCESSED_DIR, CHECKPOINT_DIR, MIDI_OUT_DIR, STEPS_PER_BEAT
from train import Vocab, MusicDataset
from model import Seq2SeqTransformer
from data_utils import event_tokens_to_notes

def notes_to_pretty_midi(notes, out_path: Path, tempo_bpm: float = 120.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=0)
    step_sec = 60.0 / tempo_bpm / STEPS_PER_BEAT

    for n in notes:
        start = n.onset * step_sec
        end = (n.onset + n.duration) * step_sec
        inst.notes.append(pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=float(start),
            end=float(end),
        ))

    pm.instruments.append(inst)
    pm.write(str(out_path))

@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_chords", action="store_true")
    args = parser.parse_args()

    MIDI_OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = Vocab(PROCESSED_DIR / "vocab.json")
    ds = MusicDataset(PROCESSED_DIR / "test.jsonl", vocab, use_chords=args.use_chords)

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

    for i in range(min(10, len(ds))):
        ex = ds[i]
        src = torch.tensor([ex["src_ids"]], dtype=torch.long, device=device)
        pred_ids = model.greedy_decode(src, vocab.bos_id, vocab.eos_id, max_len=1024)[0].tolist()

        pred_toks = []
        for tid in pred_ids[1:]:
            tok = vocab.itos.get(tid, "<UNK>")
            if tok == "<EOS>":
                break
            if tok not in {"<PAD>", "<BOS>"}:
                pred_toks.append(tok)

        pred_notes = event_tokens_to_notes(pred_toks)
        out_path = MIDI_OUT_DIR / f"{'chords' if args.use_chords else 'melody'}_{i:03d}.mid"
        notes_to_pretty_midi(pred_notes, out_path)

    print("[OK] wrote generated MIDI demos")

if __name__ == "__main__":
    main()