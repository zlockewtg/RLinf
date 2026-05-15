# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPi-Comet is a framework for training and deploying π0.5 (Pi05) Vision-Language-Action models on BEHAVIOR-1K robotic manipulation tasks. It supports both JAX/Flax and PyTorch backends with distributed training.

## Commands

### Setup
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync        # Install dependencies
uv pip install -e .                    # Editable install
source .venv/bin/activate              # Activate venv
pre-commit install                     # Install git hooks
```

### Linting & Formatting
```bash
ruff check .                           # Lint
ruff format .                          # Format
```
Ruff config: line-length=120, target Python 3.11.

### Testing
```bash
pytest                                 # All tests
pytest -m manual                       # Manual-only tests
pytest src/openpi/transforms_test.py   # Single test file
```

### Training
```bash
# Single-GPU JAX training
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py {config_name} --exp_name={name}

# Distributed JAX training
python scripts/train_dist.py {config_name} --exp_name={name} --overwrite

# PyTorch DDP training
python scripts/train_pytorch.py {config_name} --exp_name={name}

# Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name {config_name}
```

### Serving
```bash
uv run scripts/serve_b1k.py --task_name=$TASK policy:checkpoint --policy.config=pi05_b1k-base --policy.dir=$CKPT_PATH
```

## Architecture

### Dual-Framework Design
- `src/openpi/models/` — JAX/Flax model implementations (Pi0, Pi05, SigLIP, Gemma, LoRA)
- `src/openpi/models_pytorch/` — PyTorch equivalents for inference/training
- Checkpoints are cross-compatible; framework is auto-detected by presence of `model.safetensors`

### Configuration System
All training configs live in `src/openpi/training/config.py` as a `_CONFIGS` list. Each config is a `TrainConfig` dataclass composed of model, data, optimizer, and checkpoint sub-configs. CLI parsing uses `tyro`. To add a new config, append to `_CONFIGS`.

### Data Pipeline
```
Raw Data → RepackTransform → DataTransforms → Normalize → ModelTransforms → Model
Model Output → Unnormalize → InverseDataTransforms → Actions
```
Transforms are composable via `transforms.Group` in `src/openpi/transforms.py`. Data loaders in `src/openpi/training/data_loader.py` support multi-dataset sharding and weighted sampling.

### Policy Layer
`src/openpi/policies/` wraps models with input/output transforms for end-to-end inference. `B1KPolicyWrapper` handles BEHAVIOR-1K-specific observation processing (multi-camera, depth, point clouds).

### Key Entry Points
- `scripts/train.py` — JAX single/multi-GPU training loop
- `scripts/train_dist.py` — JAX distributed (multi-node) training
- `scripts/train_pytorch.py` — PyTorch DDP training
- `scripts/serve_b1k.py` / `scripts/serve_policy.py` — WebSocket policy servers (port 8000)
- `data_generation/rft/` — RFT data generation from rollouts

### Checkpoint Structure
Checkpoints contain `params/`, `train_state/`, and `assets/` (normalization stats). Managed by Orbax. Stored under the experiment directory.
