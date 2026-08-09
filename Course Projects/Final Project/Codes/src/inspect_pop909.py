from pathlib import Path
from collections import Counter
from config import RAW_DATA_DIR

def main():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"POP909 root not found: {RAW_DATA_DIR}")

    song_dirs = sorted([p for p in RAW_DATA_DIR.iterdir() if p.is_dir()])
    print(f"Found {len(song_dirs)} song folders")

    ext_counter = Counter()
    name_counter = Counter()

    for song_dir in song_dirs[:20]:
        print(f"\n=== {song_dir.name} ===")
        for f in sorted(song_dir.rglob("*")):
            if f.is_file():
                ext_counter[f.suffix.lower()] += 1
                name_counter[f.name] += 1
                print(" ", f.relative_to(song_dir))

    print("\nTop extensions:")
    for k, v in ext_counter.most_common():
        print(k, v)

    print("\nCommon file names:")
    for k, v in name_counter.most_common(20):
        print(k, v)

if __name__ == "__main__":
    main()