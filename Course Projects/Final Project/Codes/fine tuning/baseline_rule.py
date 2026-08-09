import json
from pathlib import Path
import pretty_midi

from config import PROCESSED_DIR, MIDI_OUT_DIR, STEPS_PER_BEAT, STEPS_PER_BAR
from data_utils import (
    event_tokens_to_notes, chord_to_pitch_classes, NoteEvent
)

def choose_chord_for_step(raw_chords, step: int) -> str:
    for c in raw_chords:
        if c["start"] <= step < c["end"]:
            return c["label"]
    return "N"

def pc_to_pitch_near(pc: int, around_pitch: int) -> int:
    candidates = [p for p in range(36, 84) if p % 12 == pc]
    return min(candidates, key=lambda p: abs(p - around_pitch))

def build_rule_based_full_piano(melody_notes, raw_chords):
    """
    Strategy:
    - Left hand: root on beat 1 and 3, root+fifth arpeggio on 2 and 4
    - Right hand: simple triad under melody every beat
    """
    out = list(melody_notes)  # full piano output includes melody

    max_step = 0
    if melody_notes:
        max_step = max(n.onset + n.duration for n in melody_notes)

    for step in range(0, max_step, STEPS_PER_BEAT):
        label = choose_chord_for_step(raw_chords, step)
        pcs = chord_to_pitch_classes(label)
        if len(pcs) == 0:
            continue

        # nearest melody note around this beat
        local_mel = [n for n in melody_notes if abs(n.onset - step) <= STEPS_PER_BEAT]
        mel_pitch = local_mel[0].pitch if local_mel else 67

        root_pc = pcs[0]
        fifth_pc = pcs[2] if len(pcs) >= 3 else pcs[0]

        beat_in_bar = (step % STEPS_PER_BAR) // STEPS_PER_BEAT

        # Left hand
        if beat_in_bar in [0, 2]:
            out.append(NoteEvent(step, pc_to_pitch_near(root_pc, 43), STEPS_PER_BEAT, 70))
        else:
            out.append(NoteEvent(step, pc_to_pitch_near(root_pc, 43), STEPS_PER_BEAT, 65))
            out.append(NoteEvent(step, pc_to_pitch_near(fifth_pc, 50), STEPS_PER_BEAT, 60))

        # Right hand triad below melody
        base = mel_pitch - 5
        for pc in pcs[:3]:
            p = pc_to_pitch_near(pc, base)
            if p < mel_pitch + 3:
                out.append(NoteEvent(step, p, STEPS_PER_BEAT, 55))

    out.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    return out

def notes_to_pretty_midi(notes, out_path: Path, tempo_bpm: float = 120.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=0)

    step_sec = 60.0 / tempo_bpm / STEPS_PER_BEAT
    for n in notes:
        start = n.onset * step_sec
        end = (n.onset + n.duration) * step_sec
        inst.notes.append(
            pretty_midi.Note(
                velocity=n.velocity,
                pitch=n.pitch,
                start=float(start),
                end=float(end),
            )
        )
    pm.instruments.append(inst)
    pm.write(str(out_path))

def main():
    MIDI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_path = PROCESSED_DIR / "test.jsonl"

    with test_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            melody_notes = event_tokens_to_notes(ex["src_tokens"])
            pred_notes = build_rule_based_full_piano(melody_notes, ex["raw_chords"])
            out_path = MIDI_OUT_DIR / f"baseline_{idx:03d}.mid"
            notes_to_pretty_midi(pred_notes, out_path)
            if idx == 9:
                break

    print("[OK] wrote baseline MIDI demos")

if __name__ == "__main__":
    main()