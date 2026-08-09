from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pretty_midi
import re

from config import (
    BEATS_PER_BAR,
    STEPS_PER_BEAT,
    STEPS_PER_BAR,
    MAX_DURATION_STEPS,
    VELOCITY_BINS,
)

@dataclass
class NoteEvent:
    onset: int
    pitch: int
    duration: int
    velocity: int

@dataclass
class ChordEvent:
    start_step: int
    end_step: int
    label: str

ROOT_TO_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

def velocity_to_bin(v: int) -> int:
    for i in range(len(VELOCITY_BINS) - 1):
        if VELOCITY_BINS[i] <= v < VELOCITY_BINS[i + 1]:
            return i
    return len(VELOCITY_BINS) - 2

def bin_to_velocity(bin_id: int) -> int:
    bin_id = max(0, min(bin_id, len(VELOCITY_BINS) - 2))
    return (VELOCITY_BINS[bin_id] + VELOCITY_BINS[bin_id + 1] - 1) // 2

def get_beats(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    beats = pm.get_beats()
    if len(beats) < 2:
        beat_len = 0.5
        end_t = pm.get_end_time()
        beats = np.arange(0, max(end_t + beat_len, beat_len), beat_len)
    return beats

def time_to_step(t: float, beats: np.ndarray) -> int:
    if len(beats) == 0:
        return int(round(t / 0.125))  # fallback

    if t <= beats[0]:
        return 0

    idx = np.searchsorted(beats, t, side="right") - 1
    idx = max(0, min(idx, len(beats) - 1))

    if idx == len(beats) - 1:
        beat_len = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
        frac = (t - beats[idx]) / max(beat_len, 1e-6)
    else:
        beat_len = beats[idx + 1] - beats[idx]
        frac = (t - beats[idx]) / max(beat_len, 1e-6)

    sub = int(round(frac * STEPS_PER_BEAT))
    if sub >= STEPS_PER_BEAT:
        idx += 1
        sub = 0

    return idx * STEPS_PER_BEAT + sub

def duration_to_steps(start_t: float, end_t: float, beats: np.ndarray) -> int:
    s = time_to_step(start_t, beats)
    e = time_to_step(end_t, beats)
    return max(1, min(e - s, MAX_DURATION_STEPS))

def instrument_to_noteevents(inst: pretty_midi.Instrument, beats: np.ndarray) -> List[NoteEvent]:
    notes = []
    for n in inst.notes:
        onset = time_to_step(n.start, beats)
        duration = duration_to_steps(n.start, n.end, beats)
        notes.append(NoteEvent(onset, n.pitch, duration, n.velocity))
    notes.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    return notes

def load_notes_from_midi(midi_path: Path) -> Tuple[List[NoteEvent], np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    beats = get_beats(pm)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(instrument_to_noteevents(inst, beats))
    notes.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    return notes, beats

def discover_song_files(song_dir: Path) -> Dict[str, Optional[Path]]:
    """
    POP909 structure:
      song_dir/
        xxx.mid
        chord_midi.txt
        beat_midi.txt
        key_audio.txt
        versions/...
    """
    song_id = song_dir.name

    main_midi = song_dir / f"{song_id}.mid"
    chord_file = song_dir / "chord_midi.txt"
    beat_file = song_dir / "beat_midi.txt"

    return {
        "main_midi": main_midi if main_midi.exists() else None,
        "chords": chord_file if chord_file.exists() else None,
        "beats": beat_file if beat_file.exists() else None,
    }

def load_pop909_main_midi(main_midi_path: Path) -> Tuple[List[NoteEvent], List[NoteEvent], List[NoteEvent], np.ndarray]:
    """
    Load explicit POP909 tracks from the main MIDI:
      - MELODY
      - BRIDGE
      - PIANO

    Returns:
      melody_notes, bridge_notes, piano_notes, beats
    """
    pm = pretty_midi.PrettyMIDI(str(main_midi_path))
    beats = get_beats(pm)

    melody_notes = []
    bridge_notes = []
    piano_notes = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue

        name = inst.name.strip().upper()

        if name == "MELODY":
            melody_notes.extend(instrument_to_noteevents(inst, beats))
        elif name == "BRIDGE":
            bridge_notes.extend(instrument_to_noteevents(inst, beats))
        elif name == "PIANO":
            piano_notes.extend(instrument_to_noteevents(inst, beats))

    melody_notes.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    bridge_notes.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    piano_notes.sort(key=lambda x: (x.onset, x.pitch, x.duration))

    if len(melody_notes) == 0:
        raise ValueError(f"No MELODY track found in {main_midi_path}")
    if len(piano_notes) == 0:
        raise ValueError(f"No PIANO track found in {main_midi_path}")

    return melody_notes, bridge_notes, piano_notes, beats

def merge_notes(a: List[NoteEvent], b: List[NoteEvent]) -> List[NoteEvent]:
    out = list(a) + list(b)
    out.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    return out

def crop_notes_to_segment(notes: List[NoteEvent], start_step: int, end_step: int) -> List[NoteEvent]:
    out = []
    for n in notes:
        if start_step <= n.onset < end_step:
            dur = min(n.duration, end_step - n.onset)
            out.append(NoteEvent(n.onset - start_step, n.pitch, dur, n.velocity))
    return out

def parse_chord_label(label: str) -> Tuple[Optional[int], str]:
    label = label.strip()
    if not label or label.upper() in {"N", "NO_CHORD"}:
        return None, "none"

    # remove slash bass, extensions handled loosely
    main = label.split("/")[0]

    m = re.match(r"^([A-G][b#]?)(.*)$", main)
    if not m:
        return None, "none"

    root_txt = m.group(1)
    qual = m.group(2).lower()

    root_pc = ROOT_TO_PC.get(root_txt)
    if root_pc is None:
        return None, "none"

    if "dim" in qual:
        quality = "dim"
    elif "aug" in qual:
        quality = "aug"
    elif "sus2" in qual:
        quality = "sus2"
    elif "sus4" in qual or "sus" in qual:
        quality = "sus4"
    elif qual.startswith("m") or "min" in qual:
        quality = "min"
    else:
        quality = "maj"

    return root_pc, quality

def chord_to_pitch_classes(label: str) -> List[int]:
    root_pc, quality = parse_chord_label(label)
    if root_pc is None:
        return []

    if quality == "maj":
        ints = [0, 4, 7]
    elif quality == "min":
        ints = [0, 3, 7]
    elif quality == "dim":
        ints = [0, 3, 6]
    elif quality == "aug":
        ints = [0, 4, 8]
    elif quality == "sus2":
        ints = [0, 2, 7]
    elif quality == "sus4":
        ints = [0, 5, 7]
    else:
        ints = [0, 4, 7]

    return [(root_pc + x) % 12 for x in ints]

def parse_chord_file(chord_path: Path, beats: np.ndarray) -> List[ChordEvent]:
    """
    Expected POP909 chord_midi.txt format:
      start_time end_time chord_label
    """
    chords = []
    lines = chord_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = re.split(r"[\s,]+", line)
        if len(parts) < 3:
            continue

        try:
            start = float(parts[0])
            end = float(parts[1])
            label = parts[2]

            s = time_to_step(start, beats)
            e = max(s + 1, time_to_step(end, beats))
            chords.append(ChordEvent(s, e, label))
        except ValueError:
            continue

    chords.sort(key=lambda x: x.start_step)
    return chords

def crop_chords_to_segment(chords: List[ChordEvent], start_step: int, end_step: int) -> List[ChordEvent]:
    out = []
    for c in chords:
        if c.end_step <= start_step or c.start_step >= end_step:
            continue
        out.append(
            ChordEvent(
                max(0, c.start_step - start_step),
                min(end_step - start_step, c.end_step - start_step),
                c.label,
            )
        )
    return out

def notes_to_event_tokens(notes: List[NoteEvent]) -> List[str]:
    tokens = []
    current_bar = -1

    notes = sorted(notes, key=lambda x: (x.onset, x.pitch, x.duration))

    for n in notes:
        bar = n.onset // STEPS_PER_BAR
        pos = n.onset % STEPS_PER_BAR

        while current_bar < bar:
            tokens.append("BAR")
            current_bar += 1

        tokens.append(f"POS_{pos}")
        tokens.append(f"NOTE_{n.pitch}")
        tokens.append(f"DUR_{min(n.duration, MAX_DURATION_STEPS)}")
        tokens.append(f"VEL_{velocity_to_bin(n.velocity)}")

    return tokens

def event_tokens_to_notes(tokens: List[str]) -> List[NoteEvent]:
    notes = []
    current_bar = -1
    current_pos = 0
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok == "BAR":
            current_bar += 1
            current_pos = 0
            i += 1
            continue

        if tok.startswith("POS_"):
            try:
                current_pos = int(tok.split("_")[1])
            except Exception:
                pass
            i += 1
            continue

        if tok.startswith("NOTE_"):
            try:
                pitch = int(tok.split("_")[1])
                dur = 1
                vel = 80

                if i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
                    dur = int(tokens[i + 1].split("_")[1])
                    i += 1
                if i + 1 < len(tokens) and tokens[i + 1].startswith("VEL_"):
                    vel = bin_to_velocity(int(tokens[i + 1].split("_")[1]))
                    i += 1

                onset = max(0, current_bar) * STEPS_PER_BAR + current_pos
                notes.append(NoteEvent(onset, pitch, dur, vel))
            except Exception:
                pass

        i += 1

    notes.sort(key=lambda x: (x.onset, x.pitch, x.duration))
    return notes

def chords_to_condition_tokens(chords: List[ChordEvent], segment_bars: int) -> List[str]:
    """
    One chord token per bar.
    If multiple chords overlap a bar, choose the first overlapping one.
    """
    out = []
    for b in range(segment_bars):
        bar_start = b * STEPS_PER_BAR
        bar_end = bar_start + STEPS_PER_BAR
        label = "N"

        for c in chords:
            if c.end_step > bar_start and c.start_step < bar_end:
                label = c.label.replace(":", "_").replace("/", "_")
                break

        out.append("BAR")
        out.append(f"CHORD_{label}")

    return out