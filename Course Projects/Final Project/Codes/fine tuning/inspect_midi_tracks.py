from pathlib import Path
import pretty_midi

from config import RAW_DATA_DIR

def inspect_song(song_id="001"):
    midi_path = RAW_DATA_DIR / song_id / f"{song_id}.mid"
    if not midi_path.exists():
        raise FileNotFoundError(f"Could not find: {midi_path}")

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    print(f"Song: {song_id}")
    print(f"Number of instruments/tracks: {len(pm.instruments)}\n")

    for i, inst in enumerate(pm.instruments):
        print(f"Track {i}")
        print(f"  name       : {repr(inst.name)}")
        print(f"  program    : {inst.program}")
        print(f"  is_drum    : {inst.is_drum}")
        print(f"  note count : {len(inst.notes)}")
        if len(inst.notes) > 0:
            pitches = [n.pitch for n in inst.notes]
            print(f"  pitch range: {min(pitches)} - {max(pitches)}")
        print()

if __name__ == "__main__":
    inspect_song("001")
    inspect_song("002")
    inspect_song("003")