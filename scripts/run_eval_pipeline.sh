#!/usr/bin/env bash
# Merge a VERL FSDP checkpoint to a HF model, then run the project's
# R-Zero-aligned math eval (6 benchmarks: math500, amc23, aime24, aime25,
# minerva_math, olympiadbench) with a GPT-4o judge re-check on failures.
#
# Usage:
#   bash scripts/run_eval_pipeline.sh <CKPT_DIR> <GPU_IDX> [OUT_DIR]
#
# Example:
#   bash scripts/run_eval_pipeline.sh \
#       /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new/global_step_80 4
#
# <CKPT_DIR> must contain `actor/` with FSDP shards and a `huggingface/` subdir.
# GPU_IDX sets CUDA_VISIBLE_DEVICES (single GPU).
# OUT_DIR defaults to <CKPT_DIR>/eval.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <CKPT_DIR> <GPU_IDX> [OUT_DIR]" >&2
  exit 1
fi

CKPT_DIR="$(readlink -f "$1")"
GPU_IDX="$2"
OUT_DIR="${3:-${CKPT_DIR}/eval}"

ACTOR_DIR="${CKPT_DIR}/actor"
HF_DIR="${CKPT_DIR}/hf_merged"

REPO_ROOT="/data1/yhoon113/evo-sample"
SCRIPTS="${REPO_ROOT}/scripts"

if [[ ! -d "${ACTOR_DIR}" ]]; then
  echo "[err] missing actor dir: ${ACTOR_DIR}" >&2
  exit 1
fi
if [[ ! -d "${ACTOR_DIR}/huggingface" ]]; then
  echo "[err] missing ${ACTOR_DIR}/huggingface (config/tokenizer source)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---- env ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPU_IDX}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

# ---- 1. FSDP -> HF merge --------------------------------------------------
if [[ -f "${HF_DIR}/config.json" ]] && \
   compgen -G "${HF_DIR}/*.safetensors" > /dev/null; then
  echo "[merge] reusing existing HF model at ${HF_DIR}"
else
  echo "[merge] ${ACTOR_DIR} -> ${HF_DIR}"
  python "${SCRIPTS}/merge_fsdp_to_hf.py" \
    --ckpt_dir "${ACTOR_DIR}" \
    --out_dir  "${HF_DIR}" \
    2>&1 | tee "${LOG_DIR}/merge.log"
fi

# ---- 2. 6-benchmark math eval (math500/amc23/aime24/aime25/minerva/olympiad)
MATH_OUT="${OUT_DIR}"
echo "[eval] math benchmarks -> ${MATH_OUT}"
python "${SCRIPTS}/eval_vllm_math.py" \
  --model "${HF_DIR}" \
  --tokenizer "${HF_DIR}" \
  --config "" \
  --output_dir "${MATH_OUT}" \
  --batch_size 128 \
  --max_tokens 4096 \
  --temperature 0.0 \
  --top_p 1.0 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.85 \
  --max_model_len 8192 \
  --dtype bfloat16 \
  2>&1 | tee "${LOG_DIR}/math_eval.log"

DETAILS="${MATH_OUT}/details.jsonl"
if [[ ! -f "${DETAILS}" ]]; then
  echo "[err] details.jsonl not produced at ${DETAILS}" >&2
  exit 1
fi

# ---- 3. GPT-4o judge re-check on failures ---------------------------------
WITH_JUDGE="${MATH_OUT}/with_gpt_judge.json"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[warn] OPENAI_API_KEY not set; skipping GPT-4o re-check." >&2
else
  echo "[judge] re-checking failures with gpt-4o -> ${WITH_JUDGE}"
  python "${SCRIPTS}/gpt_judge_recheck.py" \
    --details "${DETAILS}" \
    --out "${WITH_JUDGE}" \
    --model "gpt-4o" \
    2>&1 | tee "${LOG_DIR}/gpt_recheck.log"
fi

# ---- 4. summary -----------------------------------------------------------
echo
echo "=========================================================="
echo "Checkpoint : ${CKPT_DIR}"
echo "HF merged  : ${HF_DIR}"
echo "Eval out   : ${MATH_OUT}"
echo "Logs       : ${LOG_DIR}"
echo "----------------------------------------------------------"
if [[ -f "${WITH_JUDGE}" ]]; then
  echo "[with_gpt_judge.json]"
  python - <<PY
import json
with open("${WITH_JUDGE}") as f:
    s = json.load(f)
for name, b in s.get("benchmarks", {}).items():
    print(f"  {name:14s} pass@1={b['pass_at_1']*100:6.2f}%  ({b['correct']}/{b['total']})")
if "math_avg_6" in s:
    print(f"  MATH AVG (6, +GPT judge): {s['math_avg_6']:.2f}")
gj = s.get("gpt_judge", {})
print(f"  GPT judge: calls={gj.get('calls',0)}  yes={gj.get('yes',0)}  failed={gj.get('failed',0)}")
PY
else
  echo "[summary.json] (no GPT judge applied)"
  python - <<PY
import json
with open("${MATH_OUT}/summary.json") as f:
    s = json.load(f)
for name, b in s.get("benchmarks", {}).items():
    print(f"  {name:14s} pass@1={b['pass_at_1']*100:6.2f}%  n={b['num_examples']}")
o = s.get("overall", {})
print(f"  overall        pass@1={o.get('pass_at_1',0.0)*100:6.2f}%  n={o.get('num_examples',0)}")
PY
fi
echo "=========================================================="



: <<'END_COMMENT'
# 폴더 내 모든 step (16, 32, 48, 64, 80) 순차 평가
bash scripts/run_eval_all_steps.sh \
  /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new 0

bash scripts/run_eval_all_steps_parallel.sh \
  /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new \
  0,1,2,3
END_COMMENT
