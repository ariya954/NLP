import json
import random
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    TARGET_MODE,
    SEGMENT_BARS,
    SEGMENT_STEPS,
    RANDOM_SEED,
)
from data_utils import (
    discover_song_files,
    load_pop909_main_midi,
    merge_notes,
    crop_notes_to_segment,
    parse_chord_file,
    crop_chords_to_segment,
    notes_to_event_tokens,
    chords_to_condition_tokens,
)

def song_examples(song_dir: Path) -> List[Dict]:
    files = discover_song_files(song_dir)

    if files["main_midi"] is None:
        print(f"[WARN] Skipping {song_dir.name}: missing main midi")
        return []

    try:
        melody_notes, bridge_notes, piano_notes, beats = load_pop909_main_midi(files["main_midi"])
    except Exception as e:
        print(f"[WARN] Failed loading MIDI for {song_dir.name}: {e}")
        return []

    chords = []
    if files["chords"] is not None:
        try:
            chords = parse_chord_file(files["chords"], beats)
        except Exception as e:
            print(f"[WARN] chord parse failed for {song_dir.name}: {e}")

    if TARGET_MODE == "merged":
        target_notes = merge_notes(melody_notes, piano_notes)
    elif TARGET_MODE == "accompaniment":
        target_notes = piano_notes
    else:
        raise ValueError(f"Unknown TARGET_MODE={TARGET_MODE}")

    max_step = 0
    if melody_notes:
        max_step = max(max_step, max(int(n.onset + n.duration) for n in melody_notes))
    if target_notes:
        max_step = max(max_step, max(int(n.onset + n.duration) for n in target_notes))

    num_segments = int(max_step // SEGMENT_STEPS)
    out = []

    for seg_idx in range(num_segments):
        start_step = int(seg_idx * SEGMENT_STEPS)
        end_step = int(start_step + SEGMENT_STEPS)

        mel_seg = crop_notes_to_segment(melody_notes, start_step, end_step)
        tgt_seg = crop_notes_to_segment(target_notes, start_step, end_step)
        chd_seg = crop_chords_to_segment(chords, start_step, end_step)

        if len(mel_seg) == 0 or len(tgt_seg) == 0:
            continue

        example = {
            "song_id": str(song_dir.name),
            "segment_id": int(seg_idx),
            "src_tokens": [str(tok) for tok in notes_to_event_tokens(mel_seg)],
            "tgt_tokens": [str(tok) for tok in notes_to_event_tokens(tgt_seg)],
            "chord_tokens": [str(tok) for tok in chords_to_condition_tokens(chd_seg, SEGMENT_BARS)],
            "raw_chords": [
                {
                    "start": int(c.start_step),
                    "end": int(c.end_step),
                    "label": str(c.label),
                }
                for c in chd_seg
            ],
        }

        out.append(example)

    return out

def main():
    random.seed(RANDOM_SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    song_dirs = sorted([p for p in RAW_DATA_DIR.iterdir() if p.is_dir()])
    random.shuffle(song_dirs)

    n = len(song_dirs)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    splits = {
        "train": song_dirs[:n_train],
        "valid": song_dirs[n_train:n_train + n_valid],
        "test": song_dirs[n_train + n_valid:],
    }

    for split_name, dirs in splits.items():
        out_path = PROCESSED_DIR / f"{split_name}.jsonl"
        count = 0

        with out_path.open("w", encoding="utf-8") as f:
            for song_dir in tqdm(dirs, desc=f"Preparing {split_name}"):
                examples = song_examples(song_dir)
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    count += 1

        print(f"[OK] Wrote {count} examples to {out_path}")

if __name__ == "__main__":
    main()