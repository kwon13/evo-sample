#!/bin/bash
# RQ-Evolve Evaluation Script
# 체크포인트 경로를 인자로 받아 GSM8K, MATH-500, AIME-2024 평가
#
# Usage:
#   bash scripts/run_eval.sh ./rq_output/verl_ckpt/global_step_50
#   bash scripts/run_eval.sh Qwen/Qwen3-8B-Base   # baseline 평가

MODEL_PATH=${1:?Usage: bash scripts/run_eval.sh <checkpoint_or_model_path>}
TP=${2:-2}
OUTPUT_DIR=${3:-./rq_output/eval_results}

python scripts/evaluate.py \
    --model_path "$MODEL_PATH" \
    --tp $TP \
    --gsm8k_samples -1 \
    --math500_samples -1 \
    --aime_k 32 \
    --max_tokens 8192 \
    --output_dir "$OUTPUT_DIR"
