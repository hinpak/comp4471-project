"""
Device configuration utility for cross-platform deep learning.

Automatically selects the best available device:
- CUDA (if NVIDIA GPU + CUDA Toolkit installed on Windows/Linux)
- MPS (Metal Performance Shaders on macOS with compatible GPU)
- CPU (fallback)

Usage:
    from src.device import DEVICE
    model = MyModel().to(DEVICE)
    data = data.to(DEVICE)
"""

import torch


def get_device():
    """
    Get the best available device for the current system.
    
    Returns:
        torch.device: The recommended device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Metal Performance Shaders (macOS)")
        return device
    else:
        device = torch.device('cpu')
        print("ℹ Using CPU (training will be slower)")
        return device


# Global device - automatically set on import
DEVICE = get_device()


if __name__ == "__main__":
    print(f"\nActive device: {DEVICE}")
    print(f"Device type: {DEVICE.type}")
    
    if DEVICE.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
