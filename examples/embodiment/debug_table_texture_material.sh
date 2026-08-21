#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl"
REPO_PATH="${ROOT_DIR}/RLinf"
OUTPUT_DIR="${1:-${REPO_PATH}/logs/debug_table_texture_material/$(date -u +%Y%m%d-%H%M%S)}"
ISAAC_SETUP="${REPO_PATH}/isaac_sim/setup_conda_env.sh"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
if [[ -f "${ISAAC_SETUP}" ]]; then
  set +u
  source "${ISAAC_SETUP}"
  set -u
fi

"${REPO_PATH}/.venv/bin/python" "${REPO_PATH}/examples/embodiment/debug_table_texture_material.py" \
  --output-dir "${OUTPUT_DIR}" \
  --texture-keys copper brass steel_stainless \
  --width 256 \
  --height 256 \
  --render-frames 8
