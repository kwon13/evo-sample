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

ACTOR_DIR="${CKPT_DIR}/actor"
HF_DIR="${CKPT_DIR}/hf_merged"
OUT_DIR="${CKPT_DIR}/${OUT_SUBDIR}"

REPO_ROOT="/data1/yhoon113/evo-sample"
SCRIPTS="${REPO_ROOT}/scripts"

mkdir -p "${OUT_DIR}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Ensure hf_merged exists: if not, merge from actor/. Errors out if neither
# is available.
hf_merged_present() {
  [[ -f "${HF_DIR}/config.json" ]] && compgen -G "${HF_DIR}/*.safetensors" > /dev/null
}

if ! hf_merged_present; then
  if [[ -d "${ACTOR_DIR}" && -d "${ACTOR_DIR}/huggingface" ]]; then
    echo "[merge] no hf_merged/, merging from ${ACTOR_DIR} -> ${HF_DIR}"
    python "${SCRIPTS}/merge_fsdp_to_hf.py" \
      --ckpt_dir "${ACTOR_DIR}" \
      --out_dir  "${HF_DIR}" \
      2>&1 | tee "${LOG_DIR}/merge.log"
    if ! hf_merged_present; then
      echo "[err] merge did not produce a valid HF model at ${HF_DIR}" >&2
      exit 1
    fi
  else
    echo "[err] no usable model: ${HF_DIR} missing and ${ACTOR_DIR}/huggingface not found" >&2
    exit 1
  fi
else
  echo "[merge] reusing existing HF model at ${HF_DIR}"
fi

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

# REASONING_ONLY=1 skips math eval + GPT judge; useful when math has already
# been run on these checkpoints and you only want to fill in the reasoning
# benchmarks. Implicitly enables REASONING_EVAL.
if [[ "${REASONING_ONLY:-0}" == "1" ]]; then
  REASONING_EVAL=1
  echo "[skip] REASONING_ONLY=1: skipping math eval and GPT judge"
else
  # ---- 1. 6-benchmark math eval (long context) ----------------------------
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

  # ---- 2. GPT-4o judge re-check on failures -------------------------------
  WITH_JUDGE="${OUT_DIR}/with_gpt_judge.json"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[warn] OPENAI_API_KEY not set; skipping GPT-4o re-check." >&2
  else
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
    echo "[judge] re-checking failures with ${JUDGE_MODEL} -> ${WITH_JUDGE}"
    python "${SCRIPTS}/gpt_judge_recheck.py" \
      --details "${DETAILS}" \
      --out "${WITH_JUDGE}" \
      --model "${JUDGE_MODEL}" \
      2>&1 | tee "${LOG_DIR}/gpt_recheck.log"
  fi
fi

# ---- 3. reasoning benchmarks (BBEH / MMLU-Pro / SuperGPQA) ----------------
# Same toggle as run_eval_pipeline.sh. Reasoning eval uses its own max_tokens
# (default 8192 to match R-Zero); the longlen math max_tokens does not apply.
# Output goes under <OUT_DIR>/<bench>/ so it sits alongside the longlen math
# results. Override REASONING_OUT_ROOT if you want a different location.
if [[ "${REASONING_EVAL:-0}" == "1" ]]; then
  REASONING_BENCHMARKS="${REASONING_BENCHMARKS:-bbeh,mmlupro,supergpqa}"
  REASONING_MAX_SAMPLES="${REASONING_MAX_SAMPLES:--1}"
  REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-8192}"
  REASONING_OUT_ROOT="${REASONING_OUT_ROOT:-${OUT_DIR}}"

  declare -A REASONING_SCRIPTS=(
    [bbeh]="${REPO_ROOT}/evaluation/eval_bbeh.py"
    [mmlupro]="${REPO_ROOT}/evaluation/eval_mmlupro.py"
    [supergpqa]="${REPO_ROOT}/evaluation/eval_supergpqa.py"
  )

  IFS=',' read -ra REASONING_LIST <<< "${REASONING_BENCHMARKS}"
  for bench in "${REASONING_LIST[@]}"; do
    bench="$(echo "${bench}" | tr -d '[:space:]')"
    [[ -z "${bench}" ]] && continue
    script="${REASONING_SCRIPTS[${bench}]:-}"
    if [[ -z "${script}" ]]; then
      echo "[warn] unknown reasoning benchmark '${bench}', skipping" >&2
      continue
    fi
    bench_out="${REASONING_OUT_ROOT}/${bench}"
    mkdir -p "${bench_out}"
    echo "[eval] ${bench} -> ${bench_out}  (max_samples=${REASONING_MAX_SAMPLES})"
    python "${script}" \
      --model_path "${HF_DIR}" \
      --tokenizer "${HF_DIR}" \
      --output_dir "${bench_out}" \
      --max_samples "${REASONING_MAX_SAMPLES}" \
      --max_tokens "${REASONING_MAX_TOKENS}" \
      --temperature 0.0 \
      --top_p 1.0 \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.85 \
      --max_model_len "${MAX_MODEL_LEN}" \
      --dtype bfloat16 \
      2>&1 | tee "${LOG_DIR}/${bench}_eval.log"
  done

  echo "----------------------------------------------------------"
  echo "[reasoning benchmarks]"
  python - <<PY
import json, os
out_root = "${REASONING_OUT_ROOT}"
for bench in ("bbeh", "mmlupro", "supergpqa"):
    path = os.path.join(out_root, bench, "summary.json")
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        s = json.load(f)
    b = s.get("benchmarks", {}).get(bench, {})
    line = f"  {bench:10s} pass@1={b.get('pass_at_1', 0.0)*100:6.2f}%  n={b.get('num_examples', 0)}"
    if "macro_avg" in b:
        line += f"  macro={b['macro_avg']*100:6.2f}%"
    print(line)
PY
fi
