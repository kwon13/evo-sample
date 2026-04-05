#!/bin/bash
# R_Q Evolutionary Problem Generation - Full Pipeline
# 
# Prerequisites:
#   pip install verl vllm torch transformers datasets sentence-transformers sympy pyarrow
#
# Hardware: 4x GPU (A100/H100 recommended)
#
# Usage:
#   bash scripts/run_full.sh

set -e

export STORAGE_PATH=${STORAGE_PATH:-"./rq_output"}
export WANDB_MODE=${WANDB_MODE:-"disabled"}

echo "================================================"
echo "R_Q Evolutionary Problem Generation Pipeline"
echo "================================================"

# --- Step 1: Prepare seed programs from LIMR ---
echo ""
echo "[Step 1/3] Preparing seed programs from LIMR dataset..."
python run.py prepare \
    --output_dir ./seed_programs \
    --max_static 10 \
    --max_limr 500

# --- Step 2: Run Questioner evolution ---
echo ""
echo "[Step 2/3] Running Questioner evolution..."
python run.py evolve \
    --seed_dir ./seed_programs \
    --output_dir ${STORAGE_PATH} \
    --num_epochs 3 \
    --num_generations 50 \
    --candidates_per_gen 4 \
    --num_rollouts 16 \
    --n_h_bins 6 \
    --n_div_bins 6 \
    --mutator_model Qwen/Qwen2.5-7B-Instruct \
    --solver_model Qwen/Qwen2.5-3B

# --- Step 3: Train Solver with veRL GRPO ---
echo ""
echo "[Step 3/3] Training Solver with veRL GRPO..."

# Find the latest training data
LATEST_TRAIN=$(ls -t ${STORAGE_PATH}/train_epoch_*.parquet 2>/dev/null | head -1)

if [ -z "$LATEST_TRAIN" ]; then
    LATEST_TRAIN=$(ls -t ${STORAGE_PATH}/train_epoch_*.jsonl 2>/dev/null | head -1)
fi

if [ -z "$LATEST_TRAIN" ]; then
    echo "ERROR: No training data found. Evolution may have failed."
    exit 1
fi

echo "Using training data: $LATEST_TRAIN"

# Launch veRL GRPO training
python -m verl.trainer.main_ppo \
    data.train_files=$LATEST_TRAIN \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.kl_ctrl.kl_coeff=0.001 \
    trainer.total_epochs=1 \
    trainer.project_name=rq_evolve \
    trainer.experiment_name=rq_solver \
    trainer.default_local_dir=${STORAGE_PATH}/verl_checkpoints

echo ""
echo "================================================"
echo "Pipeline complete!"
echo "Results saved to: ${STORAGE_PATH}"
echo "================================================"
