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
