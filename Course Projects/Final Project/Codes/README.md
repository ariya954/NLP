# Melody-to-Piano
**Transformer-Based Full Piano Accompaniment Generation from Simple Melody**

## Overview
this is overview

## Project Structure
```text
project/
├── data/
│   ├── raw/POP909/
│   ├── processed/
├── outputs/
│   ├── checkpoints/
│   ├── midi/
│   ├── metrics/
├── src/
│   ├── config.py
│   ├── inspect_pop909.py
│   ├── inspect_midi_tracks.py
│   ├── data_utils.py
│   ├── prepare_dataset.py
│   ├── build_vocab.py
│   ├── baseline_rule.py
│   ├── model.py
│   ├── train.py
│   ├── sanity_check.py
│   ├── evaluate.py
│   ├── generate_demo.py
├── project.ipynb
├── requirements.txt
└── README.md
```
## Environment Setup
Use Python 3.10 or 3.11 only to avoid Windows MIDI parsing issues.

```bash
cd project
python -m venv .venv
source .venv/bin/activate   # mac/linux
.venv\Scripts\activate      # windows

pip install --upgrade pip
pip install -r requirements.txt
```

\
Potential problems and fixes:
- `pretty_midi` fails: install `mido` first, then `pretty_midi`
- Torch GPU not detected: run on CPU first; training still works for debugging

## Instructions
### Step 1: Download Required Data
Download POP909 from [POP909 Dataset for Music Arrangement Generation](https://github.com/music-x-lab/POP909-Dataset.git) into:

```text
data/raw/
```

\
To see the files that have been downloaded, run:
```bash
python src/inspect_pop909.py
```

The output should at least include the following in each index folder:
- index.mid
- chord_midi.txt
- beat_midi.txt

\
To ensure the data is accurate, it should contain both `MELODY` and `PIANO`. Run:
```bash
python src/inspect_midi_tracks.py
```

The first output should resemble this:
```text
Song: 001
Number of instruments/tracks: 3

Track 0
  name       : 'MELODY'
  program    : 0
  is_drum    : False
  note count : 264
  pitch range: 61 - 70

Track 1
  name       : 'BRIDGE'
  program    : 0
  is_drum    : False
  note count : 307
  pitch range: 61 - 87

Track 2
  name       : 'PIANO'
  program    : 0
  is_drum    : False
  note count : 985
  pitch range: 39 - 70
```

---

### Step 2: Build the Core MIDI and Chord Utilities
The objective in the `src/data_utils.py` file is to implement:
- Note loading
- Beat-aware quantization
- Chord parsing
- Segment extraction
- Event token conversion

---

### Step 3: Prepare the Tokenized Dataset
For every song, the task is to:
- Find main MIDI and chord files,
- Load notes,
- Split main MIDI into melody and piano notes,
- Create 4-bar segments,
- Produce `train.jsonl`, `valid.jsonl`, `test.jsonl`, and save all of them in `data/processed/` directory.

Run:
```bash
python src/prepare_dataset.py
```

---

### Step 4: Build the Vocabulary
The main goal is to create a vocabulary using only training tokens. Then, save `vocab.json` in the `data/processed/` directory.

Run:
```bash
python src/build_vocab.py
```

---

### Step 5: Rule-Based Baseline
The task is to generate a simple accompaniment using a melody and chords. This will create `baseline_xxx.mid` files that are saved in the `outputs/midi/` directory.

Run:
```bash
python src/baseline_rule.py
```

\
To change the number of baseline examples, modify the `num_songs` variable on **line 86** in `src/baseline_rule.py` to the preferred amount.

```python
num_songs = 10
```

\
Potential problem and fix:
- Melody is duplicated awkwardly: if target mode is `merged`, this is acceptable. Otherwise, consider switching to `accompaniment`

---

### Step 6: Transformer Model
The objective of the `src/model.py` is to implement a small seq2seq Transformer with greedy decoding.

---

### Step 7: Training Pipeline
There are two components to train:
- Melody-only Transformer
- Chord-conditioned Transformer

Each Transformer will be trained until the maximum number of epochs, saving the best checkpoint. These checkpoints will be stored as `transformer_melody.pt` and `transformer_chords.pt` in the `outputs/checkpoints/` directory.

Run:
```bash
python src/train.py                # Melody only
python src/train.py --use_chords   # Chord-conditioned
```

\
Potential problems and fixes:
- Outputs only `BAR` tokens:
  - Target sequences too long
  - Learning rate too high
  - The vocabulary is too large
- GPU memory issue:
  - Reduce `BATCH_SIZE`
  - Reduce `SRC_MAX_LEN`, `TGT_MAX_LEN`

---

### Step 8: Evaluation
In this step, several metrics need to be evaluated:
- Note-level Precision / Recall / F1
- Onset Precision / Recall / F1
- Chord-tone ratio (harmonic consistency)
- Register distribution statistics (left vs. right hand balance)

Each Transformer will be evaluated using above metrics. The results will be stored as `metrics_melody.json` and `metrics_chords.json` in the `outputs/metrics/` directory.

Run:
```bash
python src/evaluate.py                # Melody only
python src/evaluate.py --use_chords   # Chord-conditioned
```

\
Potential problems and fixes:
- Onset F1 is decent but note F1 is low: duration prediction is weak, however, this is common and acceptable
- Chord-tone ratio is low: chord-conditioned model should improve this
- Metrics are all near zero: check whether token decoding is broken

---

### Step 9: Generate Demo MIDIs from Trained Models
Generate some MIDIs examples for each Transformer. These MIDIs will be stored as `melody_xxx.mid` and `chords_xxx.mid` in the `outputs/midi/` directory.

Run:
```bash
python src/generate_demo.py                # Melody only
python src/generate_demo.py --use_chords   # Chord-conditioned
```

---

## Sanity Checks before Training Full Scale
### Task A: Verify One Song Manually
At least one song must be verified to enable training. Otherwise, it will be ineffective. The script in `src/sanity_check.py` contains all of the following:
- Load one song,
- Print number of melody notes,
- Print number of piano notes,
- Print first 20 source tokens,
- Print first 20 target tokens.

### Task B: Overfit 10 Examples
Temporarily modify the `MusicDataset` class in `src/train.py` to load only 10 examples. Train the model until the training loss becomes very low. If the model cannot overfit on these 10 examples, the pipeline is broken.

Please uncomment the **line 77** for this task.
```python
self.examples = self.examples[:10]
```

In `src/config.py`, change both `NUM_EPOCHS` (**line 36**) and `PATIENCE` (**line 43**) variables to 100 for this task.
```python
NUM_EPOCHS = 100
PATIENCE = 100
```

### Task C: Listen to One Baseline MIDI and One Generated MIDI
If the generated MIDI is empty or contains only one repeated note:
- Decoding is broken, or
- Model has collapsed.