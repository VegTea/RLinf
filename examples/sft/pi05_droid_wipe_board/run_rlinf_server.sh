#!/usr/bin/env bash

# Start an RLinf openpi_pytorch pi0.5-DROID WebSocket policy server.

set -euo pipefail
# Preserve an explicit caller override before common.sh supplies its legacy
# official-checkpoint default.
CALLER_NORM_STATS_DIR="${NORM_STATS_DIR:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RLINF_BASE_CHECKPOINT_DIR="${PI05_DROID_RLINF_MODEL_PATH:-/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch_rlinf}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RLINF_BASE_CHECKPOINT_DIR}}"
BUNDLED_NORM_STATS_DIR="${CHECKPOINT_DIR}/assets/wipe_board_v1_zed196_force"
if [[ -n "${CALLER_NORM_STATS_DIR}" ]]; then
  NORM_STATS_DIR="${CALLER_NORM_STATS_DIR}"
else
  if [[ -f "${BUNDLED_NORM_STATS_DIR}/norm_stats.json" ]]; then
    NORM_STATS_DIR="${BUNDLED_NORM_STATS_DIR}"
  else
    NORM_STATS_DIR="${RLINF_BASE_CHECKPOINT_DIR}/assets/wipe_board_v1_zed196_force"
  fi
fi
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8080}"
CONTROL_FREQUENCY_HZ="${CONTROL_FREQUENCY_HZ:-15}"
PYTORCH_DEVICE="${PYTORCH_DEVICE:-cuda}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-wipe the whiteboard}"
EXTERIOR_CAMERA="${EXTERIOR_CAMERA:-right}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OBSERVATION_RECORD_DIR="${OBSERVATION_RECORD_DIR:-}"
OBSERVATION_RECORD_FPS="${OBSERVATION_RECORD_FPS:-${CONTROL_FREQUENCY_HZ}}"

if [[ "${EXTERIOR_CAMERA}" != "left" && "${EXTERIOR_CAMERA}" != "right" ]]; then
  echo "EXTERIOR_CAMERA must be left or right, got: ${EXTERIOR_CAMERA}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/model.safetensors" ]]; then
  echo "RLinf model.safetensors not found: ${CHECKPOINT_DIR}/model.safetensors" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/config.json" ]]; then
  echo "RLinf config.json not found: ${CHECKPOINT_DIR}/config.json" >&2
  exit 1
fi
if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "norm_stats.json not found: ${NORM_STATS_DIR}/norm_stats.json" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

echo "Loading ${EXTERIOR_CAMERA}-camera checkpoint before opening ws://${SERVER_HOST}:${SERVER_PORT}"
RECORD_ARGS=()
if [[ -n "${OBSERVATION_RECORD_DIR}" ]]; then
  RECORD_ARGS+=(
    --observation-record-dir "${OBSERVATION_RECORD_DIR}"
    --observation-record-fps "${OBSERVATION_RECORD_FPS}"
  )
fi
exec "${PYTHON_BIN}" -u \
  toolkits/standalone_eval_scripts/openpi/serve_pi05_droid_rlinf.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --norm-stats-dir "${NORM_STATS_DIR}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --control-frequency-hz "${CONTROL_FREQUENCY_HZ}" \
  --pytorch-device "${PYTORCH_DEVICE}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --exterior-camera "${EXTERIOR_CAMERA}" \
  "${RECORD_ARGS[@]}" \
  "$@"
