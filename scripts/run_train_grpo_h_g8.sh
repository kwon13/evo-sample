#!/bin/bash
# GRPO RQ-Evolve training: G=16, frontier band L=[0.3, 0.7], H bin range [0, 1.2].

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_HOME="${CUDA_HOME:-/data1/yhoon113/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python run_verl.py --config configs/rq_config_grpo_h_g8.yaml
