#!/bin/bash
# RQ-Evolve Training Script
# verl_project (0.3.1) 기반, OmegaConf config

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=online


cd /data/kwon113/evo-sample
python run_verl.py --config configs/rq_config.yaml
