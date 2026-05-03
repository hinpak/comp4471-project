"""
preprocess_depth.py — Build the depth-map training tensor from SHREC 2017.

Pipeline per clip:
  1. Load all {i}_depth.png              →  (N, 480, 640)  raw uint16 values (mm)
  2. Per-frame crop using bounding box   →  (N, H_crop, W_crop)  hand ROI only
  3. Resize to (DEPTH_RESIZE, DEPTH_RESIZE)  →  (N, 64, 64)
  4. Normalise depth: clip at DEPTH_MAX_MM, divide by DEPTH_MAX_MM  →  [0, 1]
  5. Temporal windowing                  →  sliding windows (T=16, stride=4)
  6. Augmentation (train only)
  7. Save to .npz

Output tensors:
  X  — (M, T, 64, 64)   depth windows  (channel dim added when fed to CNN)
  y  — (M,)             gesture label (0-indexed)

WHY 64×64?
  - The original 640×480 frame contains the FULL scene. The hand occupies a
    small ROI (provided by general_informations.txt).  We crop that ROI first
    so the network only sees the hand, then resize to 64×64.
  - 64×64 keeps the per-sample memory footprint tiny while preserving the
    spatial structure a lightweight 3D-CNN needs.  Going to 112×112 or 224×224
    is valid but multiplies VRAM and training time ~3-12×.
  - 64×64 matches the input resolution used in several published depth-based
    gesture recognition works (e.g. NVIDIA gesture with R3D).
"""
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

from config import (
    RAW_DIR, PROC_DIR, TRAIN_SPLIT_FILE, TEST_SPLIT_FILE,
    DEPTH_RESIZE, DEPTH_MAX_MM, WINDOW_T, STRIDE, VAL_FRAC,
    PROC_DEPTH_TRAIN, PROC_DEPTH_VAL, PROC_DEPTH_TEST,
)
from shrec_io import (parse_manifest, iter_clips, load_depth_frames,
                      load_bboxes, COL_LABEL14)


# ── Crop & resize helpers ────────────────────────────────────────────────────

def _expand_bbox(x: int, y: int, w: int, h: int,
                 img_w: int = 640, img_h: int = 480,
                 pad_frac: float = 0.15) -> Tuple[int, int, int, int]:
    """
    Expand the bounding box by pad_frac on each side to avoid clipping
    the hand edges.  Clamps to image boundaries.
    """
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, x2, y2


def crop_and_resize_frame(frame: np.ndarray, bbox_row: np.ndarray,
                           size: int = DEPTH_RESIZE) -> np.ndarray:
    """
    frame:    (480, 640) float32 depth
    bbox_row: [x, y, w, h]
    Returns:  (size, size) float32 normalised to [0, 1]
    """
    x, y, w, h = bbox_row
    x1, y1, x2, y2 = _expand_bbox(x, y, w, h)

    # Guard degenerate bboxes
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, 640, 480

    crop = frame[y1:y2, x1:x2]

    # Normalise depth before resizing to avoid PIL quantisation artifacts
    crop = np.clip(crop, 0, DEPTH_MAX_MM) / DEPTH_MAX_MM   # [0, 1]
    crop_img = Image.fromarray((crop * 255).astype(np.uint8))
    resized  = crop_img.resize((size, size), Image.BILINEAR)
    return np.array(resized, dtype=np.float32) / 255.0      # back to [0, 1]


def process_depth_clip(clip_path: Path, n_frames: int,
                        size: int = DEPTH_RESIZE) -> np.ndarray:
    """
    Returns (N, size, size) float32 depth sequence for one clip.
    Falls back to full-frame centre-crop when bboxes are unavailable.
    """
    raw_frames = load_depth_frames(clip_path, n_frames)   # (N, 480, 640)
    bboxes     = load_bboxes(clip_path)                    # (N, 4) or None

    processed = []
    for i, frame in enumerate(raw_frames):
        if bboxes is not None and i < len(bboxes):
            bbox = bboxes[i]
        else:
            # Fallback: centre-crop a 360×360 region then resize
            bbox = np.array([140, 60, 360, 360])

        processed.append(crop_and_resize_frame(frame, bbox, size))

    return np.stack(processed, axis=0)    # (N, size, size)


# ── Temporal windowing ───────────────────────────────────────────────────────

def sliding_windows_depth(clip: np.ndarray, T: int = WINDOW_T,
                           stride: int = STRIDE) -> List[np.ndarray]:
    """
    clip: (N, H, W)
    Returns list of (T, H, W) windows, padding short clips.
    """
    N, H, W = clip.shape
    if N < T:
        pad = np.zeros((T - N, H, W), dtype=np.float32)
        clip = np.concatenate([clip, pad], axis=0)
        N = T

    windows = []
    for start in range(0, N - T + 1, stride):
        windows.append(clip[start: start + T])
    return windows


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment_depth(window: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """
    window: (T, H, W) normalised depth
    Applies:
      - Random horizontal flip      (p = 0.5)
      - Depth noise                 (σ = 0.005 ≈ 15 mm at 3 m range)
      - Random brightness shift     (±5%)
      - Random temporal dropout     (zero-out 1 random frame)
      - Random spatial crop & pad   (simulate minor camera jitter ±5 px)
    """
    w = window.copy()
    T, H, W = w.shape

    # 1. Horizontal flip
    if rng.random() < 0.5:
        w = w[:, :, ::-1].copy()

    # 2. Depth noise
    w += rng.normal(0, 0.005, w.shape).astype(np.float32)
    w = np.clip(w, 0, 1)

    # 3. Brightness (depth scale) jitter
    scale = rng.uniform(0.95, 1.05)
    w = np.clip(w * scale, 0, 1)

    # 4. Temporal dropout
    drop_idx = rng.integers(0, T)
    w[drop_idx] = 0.0

    # 5. Spatial jitter: random shift ±jit pixels then re-crop
    jit = 4
    dx, dy = rng.integers(-jit, jit + 1), rng.integers(-jit, jit + 1)
    canvas = np.zeros_like(w)
    src_x1 = max(0,  dx);  src_x2 = min(W, W + dx)
    dst_x1 = max(0, -dx);  dst_x2 = min(W, W - dx)
    src_y1 = max(0,  dy);  src_y2 = min(H, H + dy)
    dst_y1 = max(0, -dy);  dst_y2 = min(H, H - dy)
    canvas[:, dst_y1:dst_y2, dst_x1:dst_x2] = \
        w[:, src_y1:src_y2, src_x1:src_x2]
    return canvas.astype(np.float32)


# ── Main builder ─────────────────────────────────────────────────────────────

def build_depth_split(manifest: np.ndarray, augment: bool = False,
                       n_aug: int = 3, label_col: int = COL_LABEL14,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for row, clip_path in tqdm(iter_clips(RAW_DIR, manifest),
                                total=len(manifest), desc="clips"):
        n_frames = int(row[6])           # COL_SEQ_LEN
        depth_seq = process_depth_clip(clip_path, n_frames)  # (N, 64, 64)
        label = int(row[label_col]) - 1

        for win in sliding_windows_depth(depth_seq):
            X_list.append(win)
            y_list.append(label)

            if augment:
                for _ in range(n_aug):
                    X_list.append(augment_depth(win, rng))
                    y_list.append(label)

    X = np.stack(X_list, axis=0).astype(np.float32)   # (M, T, 64, 64)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def run(use_28_class: bool = False, n_aug: int = 3):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.model_selection import StratifiedShuffleSplit

    label_col = 5 if use_28_class else COL_LABEL14

    print("=== Building depth dataset ===")
    train_manifest = parse_manifest(TRAIN_SPLIT_FILE)
    test_manifest  = parse_manifest(TEST_SPLIT_FILE)

    print("[1/3] Processing test split …")
    X_test, y_test = build_depth_split(test_manifest, augment=False,
                                        label_col=label_col)
    np.savez_compressed(PROC_DEPTH_TEST, X=X_test, y=y_test)
    print(f"  test  → X{X_test.shape}  y{y_test.shape}")

    print("[2/3] Processing train split (with augmentation) …")
    X_all, y_all = build_depth_split(train_manifest, augment=True,
                                      n_aug=n_aug, label_col=label_col)

    print("[3/3] Stratified train/val split …")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC,
                                  random_state=42)
    train_idx, val_idx = next(sss.split(X_all, y_all))

    np.savez_compressed(PROC_DEPTH_TRAIN,
                        X=X_all[train_idx], y=y_all[train_idx])
    np.savez_compressed(PROC_DEPTH_VAL,
                        X=X_all[val_idx],   y=y_all[val_idx])
    print(f"  train → X{X_all[train_idx].shape}  y{y_all[train_idx].shape}")
    print(f"  val   → X{X_all[val_idx].shape}    y{y_all[val_idx].shape}")
    print("Done.")


if __name__ == "__main__":
    run()
