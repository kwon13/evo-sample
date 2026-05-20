#!/usr/bin/env bash
# Re-run only the GPT-4o judge phase on every step that already has a
# completed math eval (details.jsonl). Skips steps whose with_gpt_judge.json
# already exists. Useful when GPT judge was skipped (missing API key, etc.)
# and you don't want to re-run the GPU eval.
#
# Usage:
#   bash scripts/run_gpt_judge_only.sh <PARENT_DIR> [MAX_WORKERS]
#
# Example:
#   # make sure OPENAI_API_KEY is set first (or have it in .env)
#   bash scripts/run_gpt_judge_only.sh \
#     /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <PARENT_DIR> [MAX_WORKERS]" >&2
  exit 1
fi

PARENT="$(readlink -f "$1")"
MAX_WORKERS="${2:-32}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"

# Auto-source .env if OPENAI_API_KEY is not in the current shell.
if [[ -z "${OPENAI_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[err] OPENAI_API_KEY is not set and ${REPO_ROOT}/.env does not define it." >&2
  exit 1
fi

cd "${REPO_ROOT}"

OK=()
SKIP=()
FAIL=()

shopt -s nullglob
for d in "${PARENT}"/global_step_*/; do
  d="${d%/}"
  n="${d##*global_step_}"
  details="${d}/eval/details.jsonl"
  out="${d}/eval/with_gpt_judge.json"
  log="${d}/eval/logs/gpt_recheck.log"
  if [[ ! -f "${details}" ]]; then
    echo "[skip] step ${n}: no details.jsonl"
    SKIP+=("${n}")
    continue
  fi
  if [[ -s "${out}" ]]; then
    echo "[skip] step ${n}: with_gpt_judge.json already exists"
    SKIP+=("${n}")
    continue
  fi
  echo "[gpt]  step ${n}  ->  ${out}"
  mkdir -p "${d}/eval/logs"
  if python "${SCRIPT_DIR}/gpt_judge_recheck.py" \
        --details "${details}" \
        --out "${out}" \
        --max_workers "${MAX_WORKERS}" \
        --model "gpt-5.4-mini" 2>&1 | tee "${log}"; then
    OK+=("${n}")
  else
    rc=${PIPESTATUS[0]}
    echo "[fail] step ${n} (rc=${rc})"
    FAIL+=("${n}")
  fi
done

echo
echo "=========================================================="
echo "GPT judge re-run done."
echo "  ok   (${#OK[@]})  : ${OK[*]:-(none)}"
echo "  skip (${#SKIP[@]}) : ${SKIP[*]:-(none)}"
echo "  fail (${#FAIL[@]}) : ${FAIL[*]:-(none)}"
echo "=========================================================="

[[ ${#FAIL[@]} -gt 0 ]] && exit 2 || exit 0
