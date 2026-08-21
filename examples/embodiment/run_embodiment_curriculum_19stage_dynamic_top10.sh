#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10"

export LOG_NAME_TAG="${LOG_NAME_TAG:-19stage-dynamic-top10-${CONFIG_NAME}}"

bash "${SCRIPT_DIR}/run_embodiment.sh" "${CONFIG_NAME}" "${ROBOT_PLATFORM:-LIBERO}"
