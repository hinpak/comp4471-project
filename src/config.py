"""
config.py — Central configuration for all dataset paths, preprocessing params,
and model hyperparameters. Import this everywhere to avoid hardcoded values.
"""
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset paths ────────────────────────────────────────────────────────────
RAW_DIR   = PROJECT_ROOT / "dataset" / "raw" / "HandGestureDataset_SHREC2017"
PROC_DIR  = PROJECT_ROOT / "dataset" / "processed"

TRAIN_SPLIT_FILE = RAW_DIR / "train_gestures.txt"
TEST_SPLIT_FILE  = RAW_DIR / "test_gestures.txt"

# ── SHREC 2017 structure constants ──────────────────────────────────────────
NUM_GESTURE_CLASSES_14 = 14
NUM_GESTURE_CLASSES_28 = 28
NUM_JOINTS             = 22          # world skeleton: (N, 66) → 22 × 3
JOINT_DIMS             = 3           # x, y, z

# ── Depth frame params ───────────────────────────────────────────────────────
DEPTH_RAW_H  = 480
DEPTH_RAW_W  = 640
DEPTH_CROP_H = 112                   # spatial crop height before resize
DEPTH_CROP_W = 112                   # spatial crop width  before resize
DEPTH_RESIZE = 64                    # final square size fed to the CNN
DEPTH_MAX_MM = 3000.0                # clip depth above this (mm in PNG)

# ── Temporal windowing ───────────────────────────────────────────────────────
WINDOW_T  = 16       # frames per training sample
STRIDE    = 4        # sliding-window stride

# ── Train / val split ────────────────────────────────────────────────────────
VAL_FRAC  = 0.15     # fraction of training clips held out as validation

# ── Processed file names ─────────────────────────────────────────────────────
PROC_SKELETON_TRAIN = PROC_DIR / "skeleton_train.npz"
PROC_SKELETON_VAL   = PROC_DIR / "skeleton_val.npz"
PROC_SKELETON_TEST  = PROC_DIR / "skeleton_test.npz"

PROC_DEPTH_TRAIN    = PROC_DIR / "depth_train.npz"
PROC_DEPTH_VAL      = PROC_DIR / "depth_val.npz"
PROC_DEPTH_TEST     = PROC_DIR / "depth_test.npz"
