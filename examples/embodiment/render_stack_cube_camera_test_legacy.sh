#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl"
REPO_PATH="${ROOT_DIR}/RLinf"
ISAAC_SETUP="${REPO_PATH}/isaac_sim/setup_conda_env.sh"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
if [[ -f "${ISAAC_SETUP}" ]]; then
  set +u
  source "${ISAAC_SETUP}"
  set -u
fi

"${REPO_PATH}/.venv/bin/python" \
  "${REPO_PATH}/examples/embodiment/render_stack_cube_all_scenarios.py" \
  --scenario-file "${REPO_PATH}/examples/embodiment/config/env/isaaclab_camera_test.jsonl" \
  --isolation shared \
  "$@"
