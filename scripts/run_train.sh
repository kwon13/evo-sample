#!/bin/bash
# RQ-Evolve Training Script
# verl_project (0.3.1) 기반, OmegaConf config

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=online
export CUDA_HOME=/data1/yhoon113/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# python run_verl.py --config configs/rq_config_grpo.yaml
# 또는
python run_verl.py --config configs/rq_config_grpo_h.yaml
