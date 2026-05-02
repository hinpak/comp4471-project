# COMP4471 Project

Deep learning pipeline for COMP4471.

## Installation

### Requirements
- Python 3.10+
- `uv` package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

### Setup

```bash
# Clone repository
git clone https://github.com/hinpak/comp4471-project.git
cd comp4471-project

# Install dependencies and create virtual environment
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

### Verify Installation

Run the environment check notebook to verify setup:

```bash
jupyter notebook environment_check.ipynb
```

This checks:
- Python version
- CUDA/GPU availability
- Package versions
- System configuration

## Development

All development should use the `uv` virtual environment created during setup. Direct Python calls will use the correct environment:

```bash
uv run python script.py
```

Or activate the virtual environment and work normally.

### Device-Agnostic Code

Use the provided `DEVICE` utility to write code that works on any setup:

```python
from src.device import DEVICE

model = MyModel().to(DEVICE)
data = data.to(DEVICE)
# Code now works on Windows/GPU, Windows/CPU, macOS, and Linux
```

The device is automatically detected on startup and selected in this priority:
1. CUDA (Windows/Linux with NVIDIA GPU + CUDA Toolkit)
2. MPS (macOS with compatible GPU)
3. CPU (fallback)

## Cross-Platform Compatibility

The project is designed to work on **Windows, macOS, and Linux**. When you run `uv sync`, PyTorch automatically selects the appropriate wheels for your platform:

- **Windows/Linux without GPU**: CPU-only wheels
- **Windows/Linux with NVIDIA GPU + CUDA Toolkit installed**: GPU-optimized wheels  
- **macOS**: CPU or Metal Performance Shaders (MPS)

## GPU Acceleration (Optional)

To enable GPU acceleration on your local machine:

### Windows/Linux with NVIDIA GPU

First, ensure CUDA Toolkit is installed. Then upgrade PyTorch to your CUDA version:

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install GPU PyTorch (replace cu121 with your CUDA version)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Find your CUDA version:**
```bash
nvidia-smi  # Look for CUDA Version row
```

**Common CUDA versions:**
- CUDA 12.1: `cu121`
- CUDA 12.0: `cu120`
- CUDA 11.8: `cu118`

### macOS

Metal Performance Shaders (GPU acceleration) is automatically used if available on compatible Apple Silicon or AMD GPUs. No additional setup required.

### Verify Setup

After setup, run the environment check to verify your configuration:
```bash
jupyter notebook environment_check.ipynb
```

If GPU is available and detected, you'll see:
- `CUDA Available: True` (on Windows/Linux with CUDA)
- GPU name and memory information
- `✓ Environment ready for GPU-accelerated deep learning`

**Note:** The code automatically detects available hardware and uses the best option (GPU > CPU). All computations work on CPU if GPU is unavailable—just slower.

├── display_gesture.m             # MATLAB显示一个序列的脚本
├── display_sequence.py           # python显示一个序列的脚本(依赖Scipy, Numpy and Matplotlib)
├── gesture_1                     # 姿势id
│   ├── finger_1                 # 手指id：单个指头（finger_1）；整个手部（finger_2）
│   │   ├── subject_1           # 参与者id
│   │   │   ├── essai_1        # 序列id
│   │   │   │   ├── 0_depth.png                  # 第1帧的深度图
│   │   │   │   ├── 2_depth.png
│   │   │   │    ...
│   │   │   │   ├── N-1_depth.png                # 第N帧的深度图
│   │   │   │   ├── general_informations.txt     # shape=(N,5)为全部N帧中手部区域所在矩形框，每一行为(i, x, y, width, height)
│   │   │   │   ├── skeletons_image.txt          # shape=(N,44)为全部N帧中2D深度图像中22个手关节的二维坐标
│   │   │   │   └── skeletons_world.txt          # shape=(N,66)为全部N帧中3D世界坐标系下22个手关节的三维坐标
│   │   │   ├── essai_2
│   │   │    ...
│   │   │   └── essai_5
│   │   ├── subject_2
│   │    ...
│   │   └── subject_20
│   └── finger_2
├── gesture_2
 ...
├── gesture_14
├── test_gestures.txt             # 训练序列的信息（1960行）格式为：id_gesture      id_finger    id_subject    id_essai    14_labels    28_labels    size_sequence
└── train_gestures.txt            # 测试序列的信息（840行）

3. 使用替代数据集
如果无法获取 SHREC'17 或 DHG-14/28，可以尝试使用以下替代数据集：
EGO HANDS Dataset
MSRA Hand Pose Dataset
Handheld Hand Pose Dataset