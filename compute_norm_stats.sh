#!/bin/bash
# Compute normalization statistics for training configs.
#
# Usage:
#   bash compute_norm_stats.sh              # Compute for all modes
#   bash compute_norm_stats.sh jax          # Compute for a specific mode
#   bash compute_norm_stats.sh pytorch_fp32 # etc.

set -euo pipefail
cd /mnt/public/xzxuan/repos/openpi-comet
source .venv/bin/activate

declare -A CONFIG_MAP=(
    [jax]="pi05_b1k-task0000_sft_local"
    [pytorch_bf16]="pi05_b1k-task0000_sft_local_pytorch_bf16"
    [pytorch_fp32]="pi05_b1k-task0000_sft_local_pytorch_fp32"
    [pytorch_mixed]="pi05_b1k-task0000_sft_local_pytorch_mixed"
)

compute_stats() {
    local mode=$1
    local config_name="${CONFIG_MAP[$mode]}"
    echo "=== Computing norm stats for ${mode} (config: ${config_name}) ==="
    uv run scripts/compute_norm_stats.py --config-name "${config_name}"
    echo "Done: ${mode}"
}

MODE="${1:-all}"

if [ "$MODE" = "all" ]; then
    for mode in jax pytorch_bf16 pytorch_fp32 pytorch_mixed; do
        compute_stats "$mode"
    done
else
    if [ -z "${CONFIG_MAP[$MODE]+x}" ]; then
        echo "Error: unknown mode '$MODE'. Use: jax, pytorch_bf16, pytorch_fp32, pytorch_mixed, or all"
        exit 1
    fi
    compute_stats "$MODE"
fi
