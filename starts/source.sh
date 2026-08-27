#!/usr/bin/env bash

# Shared runtime environment for RLinf-IsaacSim6 launchers.
# Source this file before starting Ray so Ray workers inherit these values.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Usage: source ${BASH_SOURCE[0]}" >&2
    exit 2
fi

STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${STARTS_DIR}")"
MODEL_VENV="${REPO_PATH}/.venv-model312"
SIM_VENV="${REPO_PATH}/.venv-isaacsim6"
ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-$(dirname "${REPO_PATH}")/IsaacAssets6.0-minimal}"
ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-${ASSETS_ROOT}/Assets/Isaac/6.0}"

if [[ ! -x "${MODEL_VENV}/bin/python" ]]; then
    echo "ERROR: model Python not found: ${MODEL_VENV}/bin/python" >&2
    return 2
fi
if [[ ! -x "${SIM_VENV}/bin/python" ]]; then
    echo "ERROR: Isaac Sim Python not found: ${SIM_VENV}/bin/python" >&2
    return 2
fi
if [[ ! -d "${ISAACSIM_ASSET_ROOT}" ]]; then
    echo "ERROR: Isaac Sim asset root not found: ${ISAACSIM_ASSET_ROOT}" >&2
    return 2
fi

# The model-side interpreter launches the RLinf driver and Ray workers. The
# Isaac Sim interpreter is selected explicitly by the cluster YAML.
source "${MODEL_VENV}/bin/activate"
export REPO_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export RLINF_MODEL_PYTHON="${MODEL_VENV}/bin/python"
export RLINF_ISAACSIM_PYTHON="${SIM_VENV}/bin/python"
export ISAACSIM_ASSET_ROOT
export RLINF_SCENARIO_ASSET_ROOT="${RLINF_SCENARIO_ASSET_ROOT:-${REPO_PATH}/rlinf/assets_isaaclab}"
export RLINF_NODE_RANK="${RLINF_NODE_RANK:-0}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
# H100 + NCCL 2.26 can fail during NVLS setup with CUDA error 401. The
# scheduler also forwards this value to every NVIDIA worker.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

# Prevent the legacy Isaac Sim 5.1 bootstrap path from being selected by
# examples/embodiment/run_embodiment.sh when this split installation is used.
unset ISAAC_PATH EXP_PATH CARB_APP_PATH

cd "${REPO_PATH}"
