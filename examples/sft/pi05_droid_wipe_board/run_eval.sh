#!/usr/bin/env bash

# Wait for a fully initialized WebSocket server, then evaluate ten chunks.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SERVER_URL="${SERVER_URL:-ws://127.0.0.1:8080}"
WAIT_SECONDS="${WAIT_SECONDS:-900}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/outputs/pi05_droid_eval}"

deadline=$((SECONDS + WAIT_SECONDS))
echo "Waiting up to ${WAIT_SECONDS}s for ${SERVER_URL} to finish loading..."
until "${PYTHON_BIN}" - "${SERVER_URL}" <<'PY'
import sys

from openpi_client import msgpack_numpy
from websockets.sync.client import connect

url = sys.argv[1]
try:
    with connect(url, open_timeout=3, close_timeout=1) as websocket:
        metadata = msgpack_numpy.unpackb(websocket.recv())
    print(f"Server ready: {metadata}")
except Exception:
    raise SystemExit(1)
PY
do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${SERVER_URL}." >&2
    exit 1
  fi
  sleep 5
done

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u \
  toolkits/standalone_eval_scripts/openpi/evaluate_pi05_droid_server.py \
  --server-url "${SERVER_URL}" \
  --dataset-path "${DATASET_PATH}" \
  --episode-index "${EPISODE_INDEX}" \
  --num-samples "${NUM_SAMPLES}" \
  --action-horizon 15 \
  --control-frequency-hz 15 \
  --timeout-seconds 120 \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
