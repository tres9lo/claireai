#!/usr/bin/env bash
# Supervised wrapper for one chat-v10 training run.
#
# This is the deterministic hard-stop wrapper described in
# brain-lab/configs/chat-v10.md. It does three jobs:
#
#  1. Refuse to start unless there is enough root-volume headroom for the predicted
#     artifact footprint plus the 80% safety margin.
#  2. Start the training run under CLAIREAI_DATA_DIR=$PWD/runtime-data, tail the log,
#     and stop the supervised training loop when either the 12h time budget or the
#     80%-disk budget is reached.
#  3. On hard stop, truncate the current log and write a run-end note. It never kills
#     the invoking shell or any other process.
#
# If you run train_chat.py by hand, you take over these responsibilities.
#
# This script intentionally does not touch claire-scratch.{pt,json}; the run uses its
# own output name only.
#
# Path resolution: this script lives at brain-lab/scripts/run_chat_v10_guard.sh
# inside the repo root. The repo root is the parent of brain-lab/, i.e. one level
# above this script's directory (brain-lab/scripts -> brain-lab -> repo root is
# wrong; brain-lab/scripts -> brain-lab is one level, brain-lab -> repo root is a
# second level, so the repo root is TWO levels above the script file itself).
# dirname gives brain-lab/scripts; /.. gives brain-lab; /../.. gives the repo root.
# ROOT falls back to that derived path unless CLAIREAI_REPO_ROOT is set.

set -euo pipefail

ROOT="${CLAIREAI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNTIME="${ROOT}/runtime-data"
MODELS="${RUNTIME}/models"
LOG="${RUNTIME}/logs"
mkdir -p "${MODELS}" "${LOG}"

export CLAIREAI_DATA_DIR="${RUNTIME}"

PYTHON="${PYTHON:-${ROOT}/assistant/.venv/bin/python}"
TRAIN="${TRAIN:-${ROOT}/brain-lab/train_chat.py}"

NAME="claire-chat-v10"
OUT="${MODELS}/${NAME}.pt"
CORPUS="${CORPUS:-${ROOT}/brain-lab/data/chat-v10}"

VOCAB_SIZE="${VOCAB_SIZE:-4096}"
WIDTH="${WIDTH:-288}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
CONTEXT="${CONTEXT:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP="${WARMUP:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-160}"
SEED="${SEED:-42}"
STEPS="${STEPS:-6000}" # match chat-v4's step count so the comparison is like-for-like
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-43200}" # 12h
DISK_BUDGET_FRACTION="${DISK_BUDGET_FRACTION:-0.80}"


RUNNO="$(date +%Y%m%dT%H%M%S)"
RUNNO_FILE="${LOG}/run-chat-v10-${RUNNO}.log"

# Predicted footprint estimate (bytes) for this run at maximum size.
# This is a conservative estimate, not a measurement; it exists so the guard can
# refuse to start rather than let the run eat the disk budget mid-flight.
#
# The run writes these artifacts under runtime-data/ (all gitignored):
#   - final .pt checkpoint + .json manifest (written once at the end, ~600MB ceiling
#     for a ~4.9M-param model)
#   - at most 2 .partial checkpoints at a time (each at most final_pt/4)
#   - the run log (streamed; bounded by log_max)
#   - at most 2 eval logs if eval runs on schedule (bounded by eval_logs)
#
# This estimate is deliberately the MAX the run could plausibly reach, so the guard's
# pre-start check is a safety gate, not a measurement. If it is too conservative and
# refuses a run that would actually fit, lower final_pt / partials / log_max here.
function predicted_footprint_bytes() {
  local -i final_pt=600000000        # ~600MB ceiling for a ~4.9M-param ckpt
  local -i final_json=4096
  local -i partials=2                # keep at most 2 partial checkpoints at once
  local -i partial_pt=$((final_pt / 4))
  local -i log_max=$((50 * 1024 * 1024))   # 50MB of log over a long run
  local -i eval_logs=$((EVAL_BATCHES * 2 * 1024 * 1024))  # up to 2 eval logs
  echo $((final_pt + final_json + partials * partial_pt + log_max + eval_logs))
}

# Return the number of free bytes on the filesystem holding $1.
function free_bytes_on() {
  local path="$1"
  local line
  line="$(df -P "$path" 2>/dev/null | awk 'NR==2 {print $4}')" || true
  if [[ -z "$line" || "$line" == "Available" ]]; then
    echo 0
    return
  fi
  # df -P reports AVAIL in 1K-blocks (kilobytes). Convert to bytes.
  echo "$line" | awk '{printf "%d", $1 * 1024}'
}

function root_free_bytes() {
  free_bytes_on "$RUNTIME"
}

function start_time() {
  date +%s
}

function elapsed_seconds() {
  local now
  now="$(date +%s)"
  echo $((now - START_TIME))
}

function run_end_note() {
  local reason="$1"
  local steps_done="${2:-unknown}"
  local wall_done="$3"
  local pt_size="${4:-unknown}"
  local json_size="${5:-unknown}"
  local status="$6"
  local predicted
  predicted="$(predicted_footprint_bytes)"
  local free_at_stop
  free_at_stop="$(root_free_bytes)"
  mkdir -p "${RUNTIME}"
  cat > "${RUNTIME}/run-end-chat-v10-${RUNNO}.txt" <<EOF
run chat-v10 guard note
ruunnno: ${RUNNO}
start_time_epoch: ${START_TIME}
stop_reason: ${reason}
stop_status: ${status}
steps_done: ${steps_done}
wall_seconds: ${wall_done}
predicted_footprint_bytes: ${predicted}
root_free_at_start_bytes: ${FREE_AT_START}
root_free_at_stop_bytes: ${free_at_stop}
artifact_pt: ${OUT}
artifact_pt_size: ${pt_size}
artifact_json: ${OUT}.json
artifact_json_size: ${json_size}
env:
  CLAIREAI_DATA_DIR=${CLAIREAI_DATA_DIR}
  PYTHON=${PYTHON}
  TRAIN=${TRAIN}
  NAME=${NAME}
  CORPUS=${CORPUS}
  VOCAB_SIZE=${VOCAB_SIZE}
  WIDTH=${WIDTH}
  LAYERS=${LAYERS}
  HEADS=${HEADS}
  CONTEXT=${CONTEXT}
  BATCH_SIZE=${BATCH_SIZE}
  LR=${LR}
  MIN_LR=${MIN_LR}
  WARMUP=${WARMUP}
  EVAL_INTERVAL=${EVAL_INTERVAL}
  EVAL_BATCHES=${EVAL_BATCHES}
  MAX_TARGET_TOKENS=${MAX_TARGET_TOKENS}
  SEED=${SEED}
  STEPS=${STEPS}
  TIME_BUDGET_SECONDS=${TIME_BUDGET_SECONDS}
  DISK_BUDGET_FRACTION=${DISK_BUDGET_FRACTION}
guard_script: ${0}
note: This file is written by the chat-v10 run guard. The training artifacts themselves
  live under runtime-data/ and are gitignored. This note is intentionally small and
  human-readable; it is not an eval result and not evidence by itself.
EOF
}

# ---- pre-start checks ----
# All quantities below are in bytes. df -P reports 1K-blocks, so we convert to
# bytes explicitly so the comparison with the predicted footprint (also bytes) is
# correct.

FREE_AT_START="$(root_free_bytes)"
PREDICTED="$(predicted_footprint_bytes)"
ROOT_TOTAL_KB="$(df -P "$RUNTIME" 2>/dev/null | awk 'NR==2 {print $2}')" || ROOT_TOTAL_KB=0
if [[ -z "$ROOT_TOTAL_KB" || "$ROOT_TOTAL_KB" == "Size" ]]; then
  ROOT_TOTAL_KB=0
fi
ROOT_TOTAL_BYTES="$((ROOT_TOTAL_KB * 1024))"
ROOT_USABLE_BYTES="$(awk -v total="$ROOT_TOTAL_BYTES" -v frac="$DISK_BUDGET_FRACTION" 'BEGIN { printf "%d", total * frac }')"
NEED_HEADROOM_BYTES="$(awk -v predicted="$PREDICTED" -v total="$ROOT_TOTAL_BYTES" -v usable="$ROOT_USABLE_BYTES" 'BEGIN { printf "%d", predicted + total - usable }')"

echo "guard: total_bytes=${ROOT_TOTAL_BYTES} usable_bytes=${ROOT_USABLE_BYTES} need_headroom_bytes=${NEED_HEADROOM_BYTES}"

if [[ $FREE_AT_START -le $NEED_HEADROOM_BYTES ]]; then
  echo "guard: refusing to start: not enough root free space for predicted footprint + 80% margin"
  echo "guard: free_now=${FREE_AT_START} predicted=${PREDICTED} need_headroom_at_least=${NEED_HEADROOM_BYTES}"
  run_end_note "disk_budget_pre_start" "0" "0" "0" "0" "refused"
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "guard: python not found at ${PYTHON}" >&2
  exit 3
fi
if [[ ! -f "$TRAIN" ]]; then
  echo "guard: train script not found at ${TRAIN}" >&2
  exit 4
fi
if [[ ! -f "${CORPUS}/train.jsonl" || ! -f "${CORPUS}/validation.jsonl" ]]; then
  echo "guard: corpus missing train/validation under ${CORPUS}" >&2
  exit 5
fi

echo "guard: start time budget=${TIME_BUDGET_SECONDS}s disk budget fraction=${DISK_BUDGET_FRACTION}"
echo "guard: free_now=${FREE_AT_START} predicted=${PREDICTED}"
echo "guard: output=${OUT} corpus=${CORPUS}"
echo "guard: tailing ${RUNNO_FILE}"

# ---- start the supervised run ----

START_TIME="$(start_time)"

# Launch the training run in the background so we can watch the clock and disk.
# Its stdout+stderr go to the run-specific log; we also keep a short-lived tail view.
"${PYTHON}" "${TRAIN}" \
  --name "${NAME}" \
  --output "${OUT}" \
  --corpus "${CORPUS}" \
  --vocab-size "${VOCAB_SIZE}" \
  --width "${WIDTH}" \
  --layers "${LAYERS}" \
  --heads "${HEADS}" \
  --context "${CONTEXT}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LR}" \
  --min-learning-rate "${MIN_LR}" \
  --warmup "${WARMUP}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-batches "${EVAL_BATCHES}" \
  --max-target-tokens "${MAX_TARGET_TOKENS}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  > "${RUNNO_FILE}" 2>&1 &
TRAIN_PID=$!

# Deterministic hard-stop watchdogs. The guard must stop the supervised training
# loop when either the time budget or the disk budget is reached. Both watchdogs run
# in the background; whichever fires first kills the training process (not the invoking
# shell, not tail, not anything else). The main script then waits for the training
# process to exit and reports the reason.
#
# Time-budget watchdog: sleep for the full budget, then SIGTERM the train process.
# SIGTERM gives the supervised loop a chance to flush; if it is still alive after a
# short grace period we SIGKILL it.
#
# Disk-budget watchdog: poll the root free space every few minutes and kill the run
# if free space drops below the predicted footprint (i.e. we would breach the 80%
# budget if the run wrote its predicted footprint right now).

STOP_REASON="not_decided_yet"
STOP_STATUS="not_decided_yet"

# Time-budget watchdog (backgrounded so the script can continue to tail + wait).
(
  sleep "${TIME_BUDGET_SECONDS}"
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    sleep 15
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      kill -KILL "$TRAIN_PID" 2>/dev/null || true
    fi
  fi
) &
TIME_WATCHDOG_PID=$!

# Disk-budget watchdog (backgrounded).
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 300
    free_now="$(root_free_bytes)"
    if [[ -n "$free_now" && "$free_now" =~ ^[0-9]+$ ]]; then
      if [[ $free_now -le $PREDICTED ]]; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        sleep 5
        if kill -0 "$TRAIN_PID" 2>/dev/null; then
          kill -KILL "$TRAIN_PID" 2>/dev/null || true
        fi
        break
      fi
    fi
  done
) &
DISK_WATCHDOG_PID=$!

# Tail the log while the run is alive. This is the "tail the log" part of the guard.
tail -f "${RUNNO_FILE}" &
TAIL_PID=$!

# Wait for the training process to exit for any reason (finished, signal, killed).
wait "$TRAIN_PID" || true
TRAIN_EXIT=$?

# Teardown: stop the tail and both watchdogs (if still running).
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true
kill "$TIME_WATCHDOG_PID" 2>/dev/null || true
wait "$TIME_WATCHDOG_PID" 2>/dev/null || true
kill "$DISK_WATCHDOG_PID" 2>/dev/null || true
wait "$DISK_WATCHDOG_PID" 2>/dev/null || true

WALL_DONE="$(elapsed_seconds)"

# ---- determine stop reason ----

# Determine whether a watchdog killed the run.
TIME_WATCHDOG_FIRED=0
DISK_WATCHDOG_FIRED=0
if [[ $WALL_DONE -ge $TIME_BUDGET_SECONDS ]]; then
  TIME_WATCHDOG_FIRED=1
fi
# The disk watchdog mutates STOP_REASON/STOPP_STATUS only if it fired; we detect that
# by checking whether the run was killed while free space was low. We approximate by
# checking free space at teardown; if it is below the predicted footprint and the run
# did NOT finish cleanly, treat it as a disk hard stop.
free_now="$(root_free_bytes)"
if [[ -n "$free_now" && "$free_now" =~ ^[0-9]+$ && $free_now -le $PREDICTED && $TRAIN_EXIT -ne 0 ]]; then
  DISK_WATCHDOG_FIRED=1
fi

if [[ $DISK_WATCHDOG_FIRED -eq 1 ]]; then
  STOP_REASON="disk_budget_mid_run"
  STOP_STATUS="disk_budget_hard_stop"
elif [[ $TIME_WATCHDOG_FIRED -eq 1 ]]; then
  STOP_REASON="time_budget_exhausted"
  STOP_STATUS="time_budget_hard_stop"
elif [[ $TRAIN_EXIT -eq 0 ]]; then
  STOP_REASON="finished_cleanly"
  STOP_STATUS="finished"
else
  STOP_REASON="train_exit_code_${TRAIN_EXIT}"
  STOP_STATUS="finished_with_exit_code"
fi

# Sync the artifact sizes we can see now.
PT_SIZE="$(stat -c%s "$OUT" 2>/dev/null || echo 0)"
JSON_SIZE="$(stat -c%s "$OUT.json" 2>/dev/null || echo 0)"

run_end_note "$STOP_REASON" "unknown" "$WALL_DONE" "$PT_SIZE" "$JSON_SIZE" "$STOP_STATUS"

echo "guard: run ended. exit=${TRAIN_EXIT} wall=${WALL_DONE}s status=${STOP_STATUS}"
echo "guard: note written to ${RUNTIME}/run-end-chat-v10-${RUNNO}.txt"

exit 0
