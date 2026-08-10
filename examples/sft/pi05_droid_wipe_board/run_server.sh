#!/usr/bin/env bash

# Start the OpenPI WebSocket policy server on port 8080 by default.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_CHECKPOINT_DIR}}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8080}"
CONTROL_FREQUENCY_HZ="${CONTROL_FREQUENCY_HZ:-15}"
PYTORCH_DEVICE="${PYTORCH_DEVICE:-cuda}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-wipe the whiteboard}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${CHECKPOINT_DIR}/model.safetensors" ]]; then
  echo "OpenPI model.safetensors not found: ${CHECKPOINT_DIR}/model.safetensors" >&2
  echo "CHECKPOINT_DIR must point to an exported OpenPI PyTorch checkpoint." >&2
  exit 1
fi
if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "norm_stats.json not found: ${NORM_STATS_DIR}/norm_stats.json" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES
# Eager mode avoids a several-minute first-request Triton compilation. Override
# with TORCH_COMPILE_DISABLE=0 when the server will be warmed up before use.
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

echo "Loading checkpoint before opening ws://${SERVER_HOST}:${SERVER_PORT}"
exec "${PYTHON_BIN}" -u \
  toolkits/standalone_eval_scripts/openpi/serve_pi05_droid.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --norm-stats-dir "${NORM_STATS_DIR}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --control-frequency-hz "${CONTROL_FREQUENCY_HZ}" \
  --pytorch-device "${PYTORCH_DEVICE}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  "$@"
