"""
shrec_io.py — Low-level I/O helpers for the SHREC 2017 dataset.

Reads the official file layout:
  gesture_{g}/finger_{f}/subject_{s}/essai_{e}/
    {i}_depth.png          raw 16-bit depth frame
    skeletons_image.txt    (N, 44)  — 22 joints × (u, v) in pixels
    skeletons_world.txt    (N, 66)  — 22 joints × (x, y, z) in metres
    general_informations.txt (N, 5) — bounding box per frame

Split manifests:
  train_gestures.txt / test_gestures.txt
  columns: id_gesture  id_finger  id_subject  id_essai
           label_14    label_28   size_sequence
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple


# ── Column indices in train/test manifests ───────────────────────────────────
COL_GESTURE  = 0
COL_FINGER   = 1
COL_SUBJECT  = 2
COL_ESSAI    = 3
COL_LABEL14  = 4
COL_LABEL28  = 5
COL_SEQ_LEN  = 6


def parse_manifest(txt_path: Path) -> np.ndarray:
    """Return float array (N_clips, 7) from train_gestures.txt or test_gestures.txt."""
    return np.loadtxt(txt_path, dtype=int)


def clip_dir(raw_dir: Path, gesture: int, finger: int,
             subject: int, essai: int) -> Path:
    """Return the directory for one clip."""
    return (raw_dir
            / f"gesture_{gesture}"
            / f"finger_{finger}"
            / f"subject_{subject}"
            / f"essai_{essai}")


def load_skeleton_world(clip_path: Path) -> Optional[np.ndarray]:
    """
    Load 3D world-coordinate skeleton.
    Returns array of shape (N, 22, 3), or None if file is missing / empty.
    """
    f = clip_path / "skeletons_world.txt"
    if not f.exists():
        return None
    data = np.loadtxt(f)              # (N, 66)
    if data.ndim == 1:
        data = data[np.newaxis, :]    # single-frame edge case
    return data.reshape(len(data), 22, 3)


def load_skeleton_image(clip_path: Path) -> Optional[np.ndarray]:
    """
    Load 2D image-coordinate skeleton.
    Returns array of shape (N, 22, 2), or None if file is missing.
    """
    f = clip_path / "skeletons_image.txt"
    if not f.exists():
        return None
    data = np.loadtxt(f)              # (N, 44)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data.reshape(len(data), 22, 2)


def load_bboxes(clip_path: Path) -> Optional[np.ndarray]:
    """
    Load hand bounding boxes.
    Returns (N, 4) array with columns [x, y, width, height].
    (The first column 'i' is dropped.)
    """
    f = clip_path / "general_informations.txt"
    if not f.exists():
        return None
    data = np.loadtxt(f)              # (N, 5)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 1:].astype(int)    # drop frame index column


def load_depth_frames(clip_path: Path, n_frames: int) -> np.ndarray:
    """
    Load all depth PNGs for a clip.
    Returns float32 array of shape (N, H, W) with values in mm (raw PNG values).
    Missing frames are filled with zeros.
    """
    frames = []
    for i in range(n_frames):
        png = clip_path / f"{i}_depth.png"
        if png.exists():
            frames.append(np.array(Image.open(png), dtype=np.float32))
        else:
            # zero-pad missing frame — will be handled in preprocessing
            frames.append(np.zeros((480, 640), dtype=np.float32))
    return np.stack(frames, axis=0)   # (N, 480, 640)


def iter_clips(raw_dir: Path, manifest: np.ndarray):
    """
    Generator over all clips in a manifest.
    Yields (row, clip_path) for each clip where the directory exists.
    """
    for row in manifest:
        g, f, s, e = int(row[COL_GESTURE]), int(row[COL_FINGER]), \
                     int(row[COL_SUBJECT]), int(row[COL_ESSAI])
        path = clip_dir(raw_dir, g, f, s, e)
        if path.is_dir():
            yield row, path
        # silently skip missing clips (dataset has some gaps)
