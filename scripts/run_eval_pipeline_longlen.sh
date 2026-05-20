#!/usr/bin/env bash
# Long-context variant of run_eval_pipeline.sh.
#
# Evaluates a checkpoint directly from its `hf_merged/` HF model -- no actor/
# FSDP merge required, so it still works after the actor/ shards have been
# deleted. Uses a configurable max_tokens (default 24576) and writes to a
# separate output subdir (default `eval_longlen/`) so the original `eval/`
# (max_tokens 8192) results are preserved. Then runs the GPT-4o judge re-check.
#
# Runs the full R-Zero-aligned 6-benchmark suite
# (math500, amc23, aime24, aime25, minerva_math, olympiadbench).
#
# Usage:
#   bash scripts/run_eval_pipeline_longlen.sh <CKPT_DIR> <GPU_IDX> [MAX_TOKENS] [OUT_SUBDIR]
#
# Example:
#   conda activate azr-bw   # env with vllm 0.19.1
#   bash scripts/run_eval_pipeline_longlen.sh \
#       /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_org_map_pobs=8/global_step_224 0
#
# <CKPT_DIR> must contain `hf_merged/` with config.json + *.safetensors.
# GPU_IDX sets CUDA_VISIBLE_DEVICES (single GPU).
# max_model_len = MAX_TOKENS + 2048 (prompt headroom), capped at the model's
# context limit (32768 for Qwen3).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <CKPT_DIR> <GPU_IDX> [MAX_TOKENS] [OUT_SUBDIR]" >&2
  exit 1
fi

CKPT_DIR="$(readlink -f "$1")"
GPU_IDX="$2"
MAX_TOKENS="${3:-24576}"
OUT_SUBDIR="${4:-eval_longlen}"

PROMPT_HEADROOM=2048
MODEL_CTX_CAP=32768
MAX_MODEL_LEN=$(( MAX_TOKENS + PROMPT_HEADROOM ))
if (( MAX_MODEL_LEN > MODEL_CTX_CAP )); then
  MAX_MODEL_LEN="${MODEL_CTX_CAP}"
fi

HF_DIR="${CKPT_DIR}/hf_merged"
OUT_DIR="${CKPT_DIR}/${OUT_SUBDIR}"

REPO_ROOT="/data1/yhoon113/evo-sample"
SCRIPTS="${REPO_ROOT}/scripts"

if [[ ! -f "${HF_DIR}/config.json" ]] || ! compgen -G "${HF_DIR}/*.safetensors" > /dev/null; then
  echo "[err] missing HF model at ${HF_DIR} (need config.json + *.safetensors)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---- env ------------------------------------------------------------------
# Auto-source .env so the GPT-4o judge has OPENAI_API_KEY.
if [[ -z "${OPENAI_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDX}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

# ---- 1. 6-benchmark math eval (long context) ------------------------------
echo "[eval] ${HF_DIR}"
echo "[eval] max_tokens=${MAX_TOKENS}  max_model_len=${MAX_MODEL_LEN}  -> ${OUT_DIR}"
python "${SCRIPTS}/eval_vllm_math.py" \
  --model "${HF_DIR}" \
  --tokenizer "${HF_DIR}" \
  --config "" \
  --output_dir "${OUT_DIR}" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature 0.0 \
  --top_p 1.0 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.85 \
  --max_model_len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  2>&1 | tee "${LOG_DIR}/math_eval.log"

DETAILS="${OUT_DIR}/details.jsonl"
if [[ ! -f "${DETAILS}" ]]; then
  echo "[err] details.jsonl not produced at ${DETAILS}" >&2
  exit 1
fi

# ---- 2. GPT-4o judge re-check on failures ---------------------------------
WITH_JUDGE="${OUT_DIR}/with_gpt_judge.json"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[warn] OPENAI_API_KEY not set; skipping GPT-4o re-check." >&2
else
  echo "[judge] re-checking failures with gpt-4o -> ${WITH_JUDGE}"
  python "${SCRIPTS}/gpt_judge_recheck.py" \
    --details "${DETAILS}" \
    --out "${WITH_JUDGE}" \
    --model "gpt-5.4-mini" \
    2>&1 | tee "${LOG_DIR}/gpt_recheck.log"
fi
