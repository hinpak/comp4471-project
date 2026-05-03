"""
preprocess_skeleton.py — Build the skeleton training tensor from SHREC 2017.

Pipeline per clip:
  1. Load skeletons_world.txt  →  (N, 22, 3)  raw world coords (metres)
  2. Root-normalise            →  subtract joint 0 (wrist)
  3. Scale-normalise           →  divide by max pairwise joint distance
  4. Temporal windowing        →  sliding windows of T=16 frames, stride=4
  5. Augmentation (train only) →  jitter, mirror, temporal dropout, rotation
  6. Save to .npz

Output tensors:
  X  — (M, T, 22, 3)   skeleton windows
  y  — (M,)            gesture label (0-indexed, 14-class or 28-class)
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from config import (
    RAW_DIR, PROC_DIR, TRAIN_SPLIT_FILE, TEST_SPLIT_FILE,
    NUM_JOINTS, JOINT_DIMS, WINDOW_T, STRIDE, VAL_FRAC,
    PROC_SKELETON_TRAIN, PROC_SKELETON_VAL, PROC_SKELETON_TEST,
    NUM_GESTURE_CLASSES_14,
)
from shrec_io import parse_manifest, iter_clips, load_skeleton_world, COL_LABEL14


# ── Normalisation ────────────────────────────────────────────────────────────

def root_normalise(skeleton: np.ndarray) -> np.ndarray:
    """
    skeleton: (T, 22, 3)
    Subtract joint-0 (wrist) from all joints so the skeleton is
    translation-invariant.  Returns same shape.
    """
    wrist = skeleton[:, 0:1, :]          # (T, 1, 3)  — broadcast over joints
    return skeleton - wrist


def scale_normalise(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    skeleton: (T, 22, 3)  already root-normalised
    Scale by the maximum pairwise distance in the FIRST frame so that
    the gesture is size-invariant.  Returns (normalised_skeleton, scale_factor).
    """
    first = skeleton[0]                  # (22, 3)
    dists = np.linalg.norm(
        first[:, np.newaxis, :] - first[np.newaxis, :, :], axis=-1
    )                                    # (22, 22)
    scale = dists.max()
    if scale < 1e-6:
        scale = 1.0                      # degenerate clip guard
    return skeleton / scale, scale


def normalise_clip(skeleton: np.ndarray) -> np.ndarray:
    """Full normalisation pipeline: root → scale. Returns (T, 22, 3)."""
    s = root_normalise(skeleton)
    s, _ = scale_normalise(s)
    return s.astype(np.float32)


# ── Temporal windowing ───────────────────────────────────────────────────────

def sliding_windows(clip: np.ndarray, T: int = WINDOW_T,
                    stride: int = STRIDE) -> List[np.ndarray]:
    """
    clip: (N, 22, 3)
    Returns list of (T, 22, 3) windows.
    Clips shorter than T are padded with the last frame.
    """
    N = len(clip)
    if N < T:
        pad = np.tile(clip[-1:], (T - N, 1, 1))
        clip = np.concatenate([clip, pad], axis=0)
        N = T

    windows = []
    for start in range(0, N - T + 1, stride):
        windows.append(clip[start: start + T])
    return windows


# ── Data Augmentation ────────────────────────────────────────────────────────

def augment_skeleton(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    window: (T, 22, 3)
    Applies:
      - Gaussian joint noise     (σ = 0.01)
      - Random horizontal flip   (p = 0.5)
      - Random axial rotation    (±15°, around Y-axis)
      - Random temporal dropout  (zero-out 1 random frame)
    """
    w = window.copy()

    # 1. Gaussian noise
    w += rng.normal(0, 0.01, w.shape).astype(np.float32)

    # 2. Horizontal mirror — negate X coordinate
    if rng.random() < 0.5:
        w[..., 0] = -w[..., 0]

    # 3. Random Y-axis rotation
    angle = rng.uniform(-np.pi / 12, np.pi / 12)   # ±15°
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, 0, sin_a],
                  [0,     1, 0    ],
                  [-sin_a,0, cos_a]], dtype=np.float32)
    w = (R @ w.reshape(-1, 3).T).T.reshape(WINDOW_T, NUM_JOINTS, JOINT_DIMS)

    # 4. Random frame dropout (zero-out 1 frame)
    drop_idx = rng.integers(0, WINDOW_T)
    w[drop_idx] = 0.0

    return w


# ── Main builder ─────────────────────────────────────────────────────────────

def build_skeleton_split(manifest: np.ndarray, augment: bool = False,
                          n_aug: int = 3, label_col: int = COL_LABEL14,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all clips in manifest and return (X, y).
      X: (M, T, 22, 3)
      y: (M,)  0-indexed labels
    augment=True applies n_aug random augmented copies per window (train only).
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for row, clip_path in tqdm(iter_clips(RAW_DIR, manifest),
                                total=len(manifest), desc="clips"):
        skel = load_skeleton_world(clip_path)
        if skel is None:
            continue

        label = int(row[label_col]) - 1     # convert 1-indexed → 0-indexed
        skel_norm = normalise_clip(skel)    # (N, 22, 3)

        for win in sliding_windows(skel_norm):
            X_list.append(win)
            y_list.append(label)

            if augment:
                for _ in range(n_aug):
                    X_list.append(augment_skeleton(win, rng))
                    y_list.append(label)

    X = np.stack(X_list, axis=0)            # (M, T, 22, 3)
    y = np.array(y_list, dtype=np.int64)    # (M,)
    return X, y


def run(use_28_class: bool = False, n_aug: int = 3):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.model_selection import StratifiedShuffleSplit
    from config import PROC_SKELETON_TRAIN, PROC_SKELETON_VAL, PROC_SKELETON_TEST

    label_col = 5 if use_28_class else COL_LABEL14   # COL_LABEL28=5, COL_LABEL14=4

    print("=== Building skeleton dataset ===")
    train_manifest = parse_manifest(TRAIN_SPLIT_FILE)
    test_manifest  = parse_manifest(TEST_SPLIT_FILE)

    # Build test set (no augmentation, no windowing augmentation)
    print("[1/3] Processing test split …")
    X_test, y_test = build_skeleton_split(test_manifest, augment=False,
                                           label_col=label_col)
    np.savez_compressed(PROC_SKELETON_TEST, X=X_test, y=y_test)
    print(f"  test  → X{X_test.shape}  y{y_test.shape}")

    # Build full training split (with augmentation)
    print("[2/3] Processing train split (with augmentation) …")
    X_all, y_all = build_skeleton_split(train_manifest, augment=True,
                                         n_aug=n_aug, label_col=label_col)

    # Carve out validation with stratified split
    print("[3/3] Stratified train/val split …")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC,
                                  random_state=42)
    train_idx, val_idx = next(sss.split(X_all, y_all))

    np.savez_compressed(PROC_SKELETON_TRAIN,
                        X=X_all[train_idx], y=y_all[train_idx])
    np.savez_compressed(PROC_SKELETON_VAL,
                        X=X_all[val_idx],   y=y_all[val_idx])
    print(f"  train → X{X_all[train_idx].shape}  y{y_all[train_idx].shape}")
    print(f"  val   → X{X_all[val_idx].shape}    y{y_all[val_idx].shape}")
    print("Done.")


if __name__ == "__main__":
    run()
