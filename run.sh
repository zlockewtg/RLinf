#!/bin/bash
# Unified SFT training launcher for task-0000 (turning_on_radio).
#
# Usage:
#   bash run.sh <mode> [gpu_mode] [resume]
#
# Modes:
#   jax              — JAX training (mixed precision: fp32 weights, bf16 compute)
#   jax_skill        — JAX training with skill-level language labels
#   pytorch_mixed    — PyTorch mixed precision (matches JAX: fp32 weights, bf16 compute)
#   pytorch_fp32     — PyTorch pure float32
#   pytorch_bf16     — PyTorch pure bfloat16
#
# GPU modes:
#   prod (default)   — all 8 GPUs
#   debug            — last 4 GPUs (4,5,6,7)
#
# Resume:
#   resume           — resume from latest checkpoint (default: fresh start with --overwrite)
#
# Examples:
#   bash run.sh pytorch_fp32                # fresh start, 8 GPUs
#   bash run.sh pytorch_fp32 debug          # fresh start, 4 debug GPUs
#   bash run.sh pytorch_fp32 prod resume    # resume, 8 GPUs
#   bash run.sh jax debug resume            # resume JAX, 4 debug GPUs

set -euo pipefail

MODE="${1:-pytorch_mixed}"
GPU_MODE="${2:-prod}"
RESUME="${3:-}"

declare -A CONFIG_MAP=(
    [jax]="pi05_b1k-task0000_sft_local"
    [jax_skill]="pi05_b1k-task0000_sft_local_skill"
    [pytorch_bf16]="pi05_b1k-task0000_sft_local_pytorch_bf16"
    [pytorch_fp32]="pi05_b1k-task0000_sft_local_pytorch_fp32"
    [pytorch_mixed]="pi05_b1k-task0000_sft_local_pytorch_mixed"
)

declare -A EXP_MAP=(
    [jax]="task0000_sft_jax"
    [jax_skill]="task0000_sft_skill"
    [pytorch_bf16]="task0000_sft_pytorch_bf16"
    [pytorch_fp32]="task0000_sft_pytorch_fp32"
    [pytorch_mixed]="task0000_sft_pytorch_mixed"
)

if [ -z "${CONFIG_MAP[$MODE]+x}" ]; then
    echo "Error: unknown mode '$MODE'"
    echo "Usage: bash run.sh <jax|jax_skill|pytorch_mixed|pytorch_fp32|pytorch_bf16> [debug]"
    exit 1
fi

CONFIG="${CONFIG_MAP[$MODE]}"
EXP_NAME="${EXP_MAP[$MODE]}"

cd /mnt/public/xzxuan/repos/openpi-comet
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [ "$GPU_MODE" = "debug" ]; then
    echo "=== Debug: GPUs 4,5,6,7 | mode=$MODE ==="
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    NUM_GPUS=4
else
    echo "=== Production: all 8 GPUs | mode=$MODE ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    NUM_GPUS=8
fi

if [ "$RESUME" = "resume" ]; then
    RUN_FLAG="--resume"
    echo "  Resuming from latest checkpoint"
else
    RUN_FLAG="--overwrite"
fi

if [ "$MODE" = "jax" ] || [ "$MODE" = "jax_skill" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
    python scripts/train.py "$CONFIG" --exp_name="$EXP_NAME" $RUN_FLAG
else
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        scripts/train_pytorch.py \
        "$CONFIG" \
        --exp_name="$EXP_NAME" \
        $RUN_FLAG
fi
