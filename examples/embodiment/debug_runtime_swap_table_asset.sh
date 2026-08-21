#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl"
REPO_PATH="${ROOT_DIR}/RLinf"
OUTPUT_DIR="${1:-${ROOT_DIR}/RLinf/logs/debug_runtime_swap_table_asset/$(date -u +%Y%m%d-%H%M%S)}"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

"${REPO_PATH}/.venv/bin/python" "${REPO_PATH}/examples/embodiment/debug_runtime_swap_table_asset.py" \
  --output-dir "${OUTPUT_DIR}" \
  --table-assets table_copper.usd table_brass.usd \
  --num-swaps 6 \
  --width 256 \
  --height 256
