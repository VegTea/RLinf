#! /bin/bash
set -o pipefail

export PYTHONWARNINGS="ignore::FutureWarning"

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export NVIDIA_DRIVER_CAPABILITIES=all
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to the bundled Isaac Sim distribution.
export ISAAC_PATH=${ISAAC_PATH:-${REPO_PATH}/isaac_sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}
if [ -f "${ISAAC_PATH}/setup_python_env.sh" ]; then
    source "${ISAAC_PATH}/setup_python_env.sh"
else
    # Isaac Sim 6 is installed in .venv-isaacsim6. Keeping a nonexistent
    # legacy checkout path in worker environments prevents its Python package
    # from bootstrapping SimulationApp.
    unset ISAAC_PATH EXP_PATH CARB_APP_PATH
fi

if [ -z "$1" ]; then
    CONFIG_NAME="isaaclab_franka_stack_cube_ppo_openpi_pi05"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_NAME_TAG="${LOG_NAME_TAG:-${CONFIG_NAME}}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${LOG_NAME_TAG}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "${@:3}"
)
printf -v CMD_DISPLAY '%q ' "${CMD[@]}"
{
    echo "===== RLinf launch parameters ====="
    echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "hostname=$(hostname)"
    echo "pwd=$(pwd)"
    echo "config_name=${CONFIG_NAME}"
    echo "config_path=${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
    echo "log_name_tag=${LOG_NAME_TAG}"
    echo "robot_platform=${ROBOT_PLATFORM}"
    echo "libero_type=${LIBERO_TYPE}"
    echo "python=$(which python)"
    echo "repo_path=${REPO_PATH}"
    echo "log_dir=${LOG_DIR}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "nvidia_visible_devices=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia_smi_gpu_count=$(nvidia-smi -L | wc -l)"
        nvidia-smi -L | sed 's/^/gpu: /'
    else
        echo "nvidia_smi_gpu_count=<nvidia-smi not found>"
    fi
    echo "cmd=${CMD_DISPLAY}"
    echo "===== end launch parameters ====="
} > "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
exit ${PIPESTATUS[0]}
