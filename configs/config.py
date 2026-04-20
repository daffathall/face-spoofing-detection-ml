# config.py — Stable version

import os

DATA_DIR       = "data"
TRAIN_DIR      = os.path.join(DATA_DIR, "train")
TEST_DIR       = os.path.join(DATA_DIR, "test")
SAMPLE_SUB     = os.path.join(DATA_DIR, "samplesubmission.csv")
OUTPUT_DIR     = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASSES = [
    "realperson",
    "fake_printed",
    "fake_screen",
    "fake_mask",
    "fake_mannequin",
    "fake_unknown",
]
NUM_CLASSES = len(CLASSES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS   = {i: c for i, c in enumerate(CLASSES)}

SEED            = 42
NUM_EPOCHS      = 40
BATCH_SIZE      = 16   # aman untuk 4GB VRAM
GRAD_ACCUM      = 4    # effective batch = 64
NUM_WORKERS     = 4
PIN_MEMORY      = True
MIXED_PRECISION = True

N_FOLDS = 5

LR            = 1e-4
LR_DECAY      = 0.65
WEIGHT_DECAY  = 1e-3
WARMUP_EPOCHS = 5
MIN_LR        = 1e-7

EARLY_STOP = 10

MIXUP_PROB   = 0.7
MIXUP_ALPHA  = 0.4
CUTMIX_ALPHA = 1.0

# Swin Transformer — Vision Transformer, cara kerja BEDA TOTAL dari CNN
# Paling berpotensi boost ensemble karena diversity arsitektur maksimal
MODELS = [
    ("swin_tiny_patch4_window7_224", 224, "swin_t"),
]

FOCAL_GAMMA = 2.0
DROP_RATE   = 0.4
TTA_STEPS   = 5
ENSEMBLE_WEIGHTS = None
