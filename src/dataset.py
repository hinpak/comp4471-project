"""
dataset.py — PyTorch Dataset wrappers for the processed .npz files.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """
    Loads a pre-processed skeleton .npz file.
    Each item:
      X : (T, 22, 3)  float32 root+scale normalised skeleton window
      y : ()          int64 gesture label  (0-indexed)
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"])    # (M, T, 22, 3)
        self.y = torch.from_numpy(data["y"])    # (M,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DepthDataset(Dataset):
    """
    Loads a pre-processed depth .npz file.
    Each item:
      X : (1, T, H, W)  float32  — channel-first for Conv3D
      y : ()            int64 gesture label (0-indexed)
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        X = data["X"]                           # (M, T, H, W)
        self.X = torch.from_numpy(X).unsqueeze(1)  # (M, 1, T, H, W)
        self.y = torch.from_numpy(data["y"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
