#!/bin/bash
# ============================================================
# R_Q Self-Evolving Pipeline
# ============================================================
#
# One model evolves problem-generation programs (Questioner)
# and solves the generated problems (Solver) simultaneously.
#
# Prerequisites:
#   pip install -r requirements.txt
#   # or: pip install verl vllm torch transformers datasets \
#   #     sentence-transformers sympy pyarrow
#
# Hardware:
#   - Minimum: 1x A100-80GB (7B model, tp=1)
#   - Recommended: 4x A100-80GB (7B model, tp=1, faster veRL)
#   - Large scale: 8x H100 (>7B model, tp=2+)
#
# Usage:
#   # Quick test (small scale)
#   bash scripts/run_full.sh
#
#   # Custom model
#   MODEL=Qwen/Qwen2.5-3B bash scripts/run_full.sh
#
#   # Full evaluation (slower but complete)
#   EVAL_FULL=1 bash scripts/run_full.sh
#
#   # Resume from checkpoint
#   RESUME_MODEL=/path/to/checkpoint bash scripts/run_full.sh
#
#   # Enable WandB logging
#   WANDB_MODE=online WANDB_API_KEY=your_key bash scripts/run_full.sh
# ============================================================

set -euo pipefail

# --- Configuration (override via environment variables) ---

MODEL=${MODEL:-"Qwen/Qwen3-8B-Base"}
RESUME_MODEL=${RESUME_MODEL:-""}
TP=${TP:-1}
GPU_MEM=${GPU_MEM:-0.85}

NUM_EPOCHS=${NUM_EPOCHS:-5}
NUM_GENERATIONS=${NUM_GENERATIONS:-100}
CANDIDATES=${CANDIDATES:-8}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-16}
TRAIN_BATCH=${TRAIN_BATCH:-256}

OUTPUT_DIR=${OUTPUT_DIR:-"./rq_output"}
SEED_DIR=${SEED_DIR:-"./seed_programs"}

WANDB_MODE=${WANDB_MODE:-"disabled"}
export WANDB_MODE

# Evaluation: subset by default, set EVAL_FULL=1 for complete benchmarks
if [ "${EVAL_FULL:-0}" = "1" ]; then
    EVAL_GSM8K=-1       # full 1319
    EVAL_MATH500=-1     # full 500
    EVAL_AIME_K=32
else
    EVAL_GSM8K=200      # subset for fast iteration
    EVAL_MATH500=100
    EVAL_AIME_K=32
fi

# Use resume model if specified
if [ -n "$RESUME_MODEL" ]; then
    MODEL="$RESUME_MODEL"
fi

# --- Pre-flight checks ---

echo "============================================================"
echo "R_Q Self-Evolving Pipeline"
echo "============================================================"
echo "  Model:        $MODEL"
echo "  TP:           $TP"
echo "  Epochs:       $NUM_EPOCHS"
echo "  Generations:  $NUM_GENERATIONS"
echo "  Rollouts:     $NUM_ROLLOUTS"
echo "  Output:       $OUTPUT_DIR"
echo "  Eval:         GSM8K=$EVAL_GSM8K  MATH500=$EVAL_MATH500  AIME@$EVAL_AIME_K"
echo "============================================================"

# Verify seed programs exist
if [ ! -d "$SEED_DIR" ] || [ -z "$(ls -A $SEED_DIR/*.py 2>/dev/null)" ]; then
    echo "ERROR: No seed programs found in $SEED_DIR"
    exit 1
fi
echo "Seed programs: $(ls $SEED_DIR/*.py | wc -l) files"

# Verify GPU availability
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')" || {
    echo "ERROR: No CUDA GPU available"
    exit 1
}

# --- Launch pipeline ---

echo ""
echo "Starting Self-Evolving pipeline..."
echo ""

python run.py full \
    --model "$MODEL" \
    --tp "$TP" \
    --gpu_mem "$GPU_MEM" \
    --num_epochs "$NUM_EPOCHS" \
    --num_generations "$NUM_GENERATIONS" \
    --candidates_per_gen "$CANDIDATES" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --train_batch_size "$TRAIN_BATCH" \
    --eval_gsm8k "$EVAL_GSM8K" \
    --eval_math500 "$EVAL_MATH500" \
    --eval_aime_k "$EVAL_AIME_K" \
    --seed_dir "$SEED_DIR" \
    --output_dir "$OUTPUT_DIR"

# --- Summary ---

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "  Output:       $OUTPUT_DIR"
echo "  Eval results: $OUTPUT_DIR/eval/"
echo "  Champions:    $OUTPUT_DIR/final_champions/"
echo "  Log:          $OUTPUT_DIR/evolution_log.json"

if [ -f "$OUTPUT_DIR/final_summary.json" ]; then
    echo ""
    echo "Final summary:"
    python -c "
import json
with open('$OUTPUT_DIR/final_summary.json') as f:
    s = json.load(f)
print(f'  Final model:  {s.get(\"final_model\", \"N/A\")}')
gs = s.get('final_grid_stats', {})
print(f'  Coverage:     {gs.get(\"coverage\", 0):.0%}')
print(f'  Mean R_Q:     {gs.get(\"mean_rq\", 0):.4f}')
print(f'  Champions:    {gs.get(\"num_champions\", 0)}')
for e in s.get('evolution_log', []):
    ev = e.get('eval', {})
    if ev:
        parts = [f'{k}={v[\"accuracy\"]:.1%}' for k, v in ev.items()]
        print(f'  Epoch {e[\"epoch\"]}: {\"  \".join(parts)}')
"
fi

echo "============================================================"


# MODEL=Qwen/Qwen3-8B-Base WANDB_MODE=online WANDB_API_KEY=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 WANDB_PROJECT=rq_evolve TP=4 bash scripts/run_full.sh