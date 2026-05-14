#!/usr/bin/env bash
# Data-parallel across step checkpoints: run scripts/run_eval_pipeline.sh on
# every `global_step_*` subdir of <PARENT_DIR>, fanning out steps across the
# given GPU list as a worker pool. Each step occupies exactly one GPU; when a
# GPU finishes, the wrapper picks up the next step.
#
# This is data-parallel at the *step* granularity, not within a single eval.
# The eval inside each step still uses tensor_parallel_size=1 (single GPU)
# and relies on vLLM's continuous batching for intra-step throughput.
#
# Usage:
#   bash scripts/run_eval_all_steps_parallel.sh <PARENT_DIR> <GPU_LIST> [START_STEP] [END_STEP]
#
# Examples:
#   # all global_step_* across GPUs 0,1,2,3
#   bash scripts/run_eval_all_steps_parallel.sh \
#     /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new 0,1,2,3
#
#   # only steps 32..80, on GPUs 4 and 5
#   bash scripts/run_eval_all_steps_parallel.sh \
#     /data1/yhoon113/evo-sample/rq_output/verl_ckpt_grpo_h_g8_new 4,5 32 80
#
# A failure on one step is logged; the loop continues with the remaining steps.

# Intentionally NOT using `set -e`: we want the loop to survive per-step
# failures and report them at the end.
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <PARENT_DIR> <GPU_LIST> [START_STEP] [END_STEP]" >&2
  echo "       GPU_LIST = comma-separated, e.g. '0,1,2,3'" >&2
  exit 1
fi

PARENT="$(readlink -f "$1")"
GPU_LIST="$2"
START_STEP="${3:-0}"
END_STEP="${4:-999999999}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="${SCRIPT_DIR}/run_eval_pipeline.sh"

if [[ ! -f "${PIPELINE}" ]]; then
  echo "[err] pipeline script not found: ${PIPELINE}" >&2
  exit 1
fi
if [[ ! -d "${PARENT}" ]]; then
  echo "[err] parent dir not found: ${PARENT}" >&2
  exit 1
fi

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[err] empty GPU list" >&2
  exit 1
fi
for g in "${GPUS[@]}"; do
  if ! [[ "${g}" =~ ^[0-9]+$ ]]; then
    echo "[err] non-numeric GPU id: '${g}'" >&2
    exit 1
  fi
done

LOG_DIR="${PARENT}/eval_parallel_logs"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/summary.log"
: > "${SUMMARY_LOG}"

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

# ---- discover & filter step dirs -----------------------------------------
mapfile -t ALL_STEPS < <(
  find "${PARENT}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' \
    | sort -V
)

STEP_DIRS=()
SKIPPED=()
for d in "${ALL_STEPS[@]}"; do
  n="${d##*global_step_}"
  if ! [[ "${n}" =~ ^[0-9]+$ ]]; then
    SKIPPED+=("${n} (non-numeric)")
    continue
  fi
  if (( n < START_STEP )) || (( n > END_STEP )); then
    SKIPPED+=("${n} (out of [${START_STEP},${END_STEP}])")
    continue
  fi
  if [[ ! -d "${d}/actor" ]]; then
    SKIPPED+=("${n} (no actor/)")
    continue
  fi
  STEP_DIRS+=("${d}")
done

if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
  log "[err] no eligible global_step_* dirs in ${PARENT}"
  exit 1
fi

log "[plan] parent : ${PARENT}"
log "[plan] gpus   : ${GPUS[*]}"
log "[plan] steps  : $(for d in "${STEP_DIRS[@]}"; do echo -n "${d##*global_step_} "; done)"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  log "[plan] skip   : ${SKIPPED[*]}"
fi
log "[plan] logs   : ${LOG_DIR}/  (per-step file: step_<N>.log)"
log "[plan] tip    : tail -f ${LOG_DIR}/step_<N>.log to follow a step"
log ""

# ---- worker pool ----------------------------------------------------------
declare -A PID_GPU   # pid  -> gpu id
declare -A PID_STEP  # pid  -> step number
FREE_GPUS=("${GPUS[@]}")
OK_STEPS=()
FAIL_STEPS=()

cleanup() {
  log ""
  log "[abort] received signal, killing background jobs..."
  for pid in "${!PID_GPU[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}
trap cleanup INT TERM

launch_step() {
  local step_dir="$1"
  local gpu="$2"
  local n="${step_dir##*global_step_}"
  local log_file="${LOG_DIR}/step_${n}.log"
  local rc_file="${LOG_DIR}/step_${n}.rc"
  rm -f "${rc_file}"
  (
    bash "${PIPELINE}" "${step_dir}" "${gpu}"
    echo $? > "${rc_file}"
  ) >"${log_file}" 2>&1 &
  local pid=$!
  PID_GPU[${pid}]="${gpu}"
  PID_STEP[${pid}]="${n}"
  log "[$(date +%H:%M:%S)] [launch] step ${n} on gpu ${gpu}  (pid ${pid}, log: step_${n}.log)"
}

reap_one() {
  # Bash 5.0 has `wait -n` but not `-p`. Block until *any* child finishes,
  # then scan tracked PIDs to find which one is no longer alive.
  wait -n 2>/dev/null
  local finished_pid=""
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      finished_pid="${pid}"
      break
    fi
  done
  if [[ -z "${finished_pid}" ]]; then
    # No tracked PID died — likely a race or spurious wait return. Retry.
    return
  fi
  local gpu="${PID_GPU[${finished_pid}]}"
  local n="${PID_STEP[${finished_pid}]}"
  unset "PID_GPU[${finished_pid}]"
  unset "PID_STEP[${finished_pid}]"
  FREE_GPUS+=("${gpu}")

  local rc="?"
  if [[ -f "${LOG_DIR}/step_${n}.rc" ]]; then
    rc="$(cat "${LOG_DIR}/step_${n}.rc")"
  fi
  if [[ "${rc}" == "0" ]]; then
    OK_STEPS+=("${n}")
    log "[$(date +%H:%M:%S)] [done]   step ${n} on gpu ${gpu}  -> OK"
  else
    FAIL_STEPS+=("${n}")
    log "[$(date +%H:%M:%S)] [done]   step ${n} on gpu ${gpu}  -> FAIL (rc=${rc})  see ${LOG_DIR}/step_${n}.log"
  fi
}

# Dispatch loop ------------------------------------------------------------
for step_dir in "${STEP_DIRS[@]}"; do
  while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
    reap_one
  done
  gpu="${FREE_GPUS[0]}"
  FREE_GPUS=("${FREE_GPUS[@]:1}")
  launch_step "${step_dir}" "${gpu}"
done

# Drain ---------------------------------------------------------------------
while [[ ${#PID_GPU[@]} -gt 0 ]]; do
  reap_one
done

# Final summary -------------------------------------------------------------
log ""
log "=========================================================="
log "ALL STEPS DONE  (parent: ${PARENT})"
log "  ok   (${#OK_STEPS[@]})  : ${OK_STEPS[*]:-(none)}"
log "  fail (${#FAIL_STEPS[@]}) : ${FAIL_STEPS[*]:-(none)}"
log "  summary log              : ${SUMMARY_LOG}"
log "=========================================================="

if [[ ${#FAIL_STEPS[@]} -gt 0 ]]; then
  exit 2
fi
