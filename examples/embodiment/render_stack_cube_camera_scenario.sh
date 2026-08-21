#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl"
REPO_PATH="${ROOT_DIR}/RLinf"
SCENARIO_ID="${1:-000002}"
OUTPUT="${2:-${REPO_PATH}/logs/camera_scenario_${SCENARIO_ID}.png}"
ISAAC_SETUP="${REPO_PATH}/isaac_sim/setup_conda_env.sh"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
if [[ -f "${ISAAC_SETUP}" ]]; then
  set +u
  source "${ISAAC_SETUP}"
  set -u
fi

"${REPO_PATH}/.venv/bin/python" "${REPO_PATH}/examples/embodiment/render_stack_cube_camera_scenario.py" \
  --scenario-file "${REPO_PATH}/examples/embodiment/config/env/isaaclab_stack_cube.jsonl" \
  --scenario-id "${SCENARIO_ID}" \
  --output "${OUTPUT}" \
  --debug
