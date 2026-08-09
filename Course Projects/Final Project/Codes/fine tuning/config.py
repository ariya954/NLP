from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "POP909" / "POP909"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MIDI_OUT_DIR = OUTPUT_DIR / "midi"
METRICS_DIR = OUTPUT_DIR / "metrics"

# Quantization / segmentation
BEATS_PER_BAR = 4
STEPS_PER_BEAT = 4          # 16th-note grid
STEPS_PER_BAR = BEATS_PER_BAR * STEPS_PER_BEAT
SEGMENT_BARS = 4
SEGMENT_STEPS = SEGMENT_BARS * STEPS_PER_BAR

# Representation
MAX_DURATION_STEPS = 32
VELOCITY_BINS = [0, 32, 48, 64, 80, 96, 112, 128]
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SEP_CHORD_TOKEN = "<SEP_CHORDS>"

# Data mode:
# "merged" = target is melody + accompaniment together
# "accompaniment" = target is accompaniment only
TARGET_MODE = "accompaniment"

# Training
SRC_MAX_LEN = 512
TGT_MAX_LEN = 1024
BATCH_SIZE = 4
NUM_EPOCHS = 40
LR = 5e-5
D_MODEL = 256
NHEAD = 16
NUM_LAYERS = 4
FF_DIM = 1024
DROPOUT = 0.1
PATIENCE = 8
RANDOM_SEED = 42