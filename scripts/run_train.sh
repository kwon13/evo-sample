#!/bin/bash
# RQ-Evolve Training Script
# verl_project (0.3.1) 기반, OmegaConf config

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=online

# conda venv 환경 사용
PYTHON=/data/kwon113/envs/venv/bin/python

cd /data/kwon113/evo-sample
$PYTHON run_verl.py --config configs/rq_config.yaml
