#!/bin/bash
# GRPO RQ-Evolve training — fix run.
#
# Two-axis fix vs run_train_grpo_h_g8.sh:
#
#   A. CORRECTNESS (recover lost performance)
#     - configs/rq_config_grpo_h_g8_fix.yaml
#         * kl_coef: 1.0e-2 → 1.0e-3   (Hypothesis 1: policy under-trains
#                                       under 10x KL; restore 1cea28d value).
#         * experiment_name / default_local_dir suffixed with _fix so this
#           run does NOT overwrite the existing g8 archive on disk or wandb.
#     - prompts/mutation.py
#         * SCORE_FEEDBACK restores p_hat and h_score (raw values), keeps
#           rq_score hidden.  Empirical ablation (analysis/figs/hyp3_*):
#             p_hat mean        0.43 → 0.63
#             frontier fraction 60% → 29%
#             coverage          50% → 40%  at 3× more evo_idx
#         * MUTATE_CROSSOVER restores parent A/B p,H header.
#
#   B. THROUGHPUT (expected ~40~50% step-time reduction, cumulative)
#     - max_response_length: 6144 → 4096   (rollout is 70~80% of step time;
#         shorter cap = ~30~40% gen-token reduction.  CAVEAT: monitor wandb
#         response_length/mean·p95 — if responses regularly exceed 4096 the
#         truncation will eat reward; bump to 5120 in that case.)
#         max_model_len 10240 → 8192 and max_num_batched_tokens 16384 →
#         12288 follow from this.
#     - gpu_memory_utilization: 0.7 → 0.85   (more KV cache → more concurrent
#         sequences in vLLM; ~10~15% rollout win.  OOM → fall back to 0.80.)
#     - micro_batch_size_per_device_for_update: 4 → 8   (half the gradient
#         accumulation steps; ~5~10% step time.  OOM → 6.)
#     - micro_batch_size_per_device_for_experience: 2 → 4   (forward-only,
#         safe; ~3~5%.)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python run_verl.py --config configs/rq_config_grpo_h_g8_fix.yaml
