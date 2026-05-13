#!/usr/bin/env bash
# Iterate every `global_step_*` subdirectory under a verl checkpoint root and
# run scripts/run_eval_pipeline.sh on each, in ascending step order.
#
# Usage:
#   bash scripts/run_eval_all_steps.sh <PARENT_DIR> <GPU_IDX> [START_STEP] [END_STEP]
#
# Example (evaluate every step in verl_ckpt_grpo_h_g8_new on GPU 4):
#   bash scripts/run_eval_all_steps.sh \
#     /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new 4
#
# Example (only steps 32..80):
#   bash scripts/run_eval_all_steps.sh \
#     /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new 4 32 80
#
# A failure on one step is logged and the loop continues to the next step.

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <PARENT_DIR> <GPU_IDX> [START_STEP] [END_STEP]" >&2
  exit 1
fi

PARENT="$(readlink -f "$1")"
GPU_IDX="$2"
START_STEP="${3:-0}"
END_STEP="${4:-999999999}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="${SCRIPT_DIR}/run_eval_pipeline.sh"

if [[ ! -x "${PIPELINE}" && ! -f "${PIPELINE}" ]]; then
  echo "[err] pipeline script not found: ${PIPELINE}" >&2
  exit 1
fi
if [[ ! -d "${PARENT}" ]]; then
  echo "[err] parent dir not found: ${PARENT}" >&2
  exit 1
fi

# Collect step dirs, sorted by step number (natural sort).
mapfile -t STEP_DIRS < <(
  find "${PARENT}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' \
    | sort -V
)

if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
  echo "[err] no 'global_step_*' subdirs under ${PARENT}" >&2
  exit 1
fi

SUMMARY_LOG="${PARENT}/eval_all_steps.log"
: > "${SUMMARY_LOG}"

OK_STEPS=()
FAILED_STEPS=()
SKIPPED_STEPS=()

for STEP_DIR in "${STEP_DIRS[@]}"; do
  STEP_NUM="${STEP_DIR##*global_step_}"
  if ! [[ "${STEP_NUM}" =~ ^[0-9]+$ ]]; then
    echo "[skip] non-numeric step suffix: ${STEP_DIR}" | tee -a "${SUMMARY_LOG}"
    SKIPPED_STEPS+=("${STEP_NUM}")
    continue
  fi
  if (( STEP_NUM < START_STEP )) || (( STEP_NUM > END_STEP )); then
    echo "[skip] step ${STEP_NUM} outside [${START_STEP}, ${END_STEP}]" \
      | tee -a "${SUMMARY_LOG}"
    SKIPPED_STEPS+=("${STEP_NUM}")
    continue
  fi
  if [[ ! -d "${STEP_DIR}/actor" ]]; then
    echo "[skip] step ${STEP_NUM}: no actor/ subdir" | tee -a "${SUMMARY_LOG}"
    SKIPPED_STEPS+=("${STEP_NUM}")
    continue
  fi

  echo "==========================================================" \
    | tee -a "${SUMMARY_LOG}"
  echo "[loop] step ${STEP_NUM}: ${STEP_DIR}" | tee -a "${SUMMARY_LOG}"
  echo "==========================================================" \
    | tee -a "${SUMMARY_LOG}"

  if bash "${PIPELINE}" "${STEP_DIR}" "${GPU_IDX}"; then
    OK_STEPS+=("${STEP_NUM}")
    echo "[ok] step ${STEP_NUM}" | tee -a "${SUMMARY_LOG}"
  else
    rc=$?
    FAILED_STEPS+=("${STEP_NUM}")
    echo "[fail] step ${STEP_NUM} (exit ${rc}) — continuing" \
      | tee -a "${SUMMARY_LOG}"
  fi
done

echo | tee -a "${SUMMARY_LOG}"
echo "==========================================================" \
  | tee -a "${SUMMARY_LOG}"
echo "ALL STEPS DONE" | tee -a "${SUMMARY_LOG}"
echo "  parent     : ${PARENT}" | tee -a "${SUMMARY_LOG}"
echo "  ok    (${#OK_STEPS[@]}) : ${OK_STEPS[*]:-(none)}" | tee -a "${SUMMARY_LOG}"
echo "  fail  (${#FAILED_STEPS[@]}) : ${FAILED_STEPS[*]:-(none)}" \
  | tee -a "${SUMMARY_LOG}"
echo "  skip  (${#SKIPPED_STEPS[@]}) : ${SKIPPED_STEPS[*]:-(none)}" \
  | tee -a "${SUMMARY_LOG}"
echo "  log        : ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==========================================================" \
  | tee -a "${SUMMARY_LOG}"

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  exit 2
fi
