#!/usr/bin/env bash

# Train the JAX-aligned RLinf OpenPI implementation on wipe-board DROID data.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export EMBODIED_PATH="${REPO_ROOT}/examples/sft"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PI05_DROID_RLINF_MODEL_PATH="${PI05_DROID_RLINF_MODEL_PATH:-/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch_rlinf}"
export PI05_DROID_RLINF_NORM_STATS="${PI05_DROID_RLINF_NORM_STATS:-${PI05_DROID_RLINF_MODEL_PATH}/assets/wipe_board_v1_zed196_force}"

RUN_ROOT="${SFT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/pi05_droid_openpi_pytorch_sft}"
RUN_LOG_DIR="${RUN_ROOT}/logs/$(date -u +'%Y%m%d-%H%M%S')"
mkdir -p "${RUN_LOG_DIR}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" examples/sft/train_vla_sft.py \
  --config-path "${REPO_ROOT}/examples/sft/config" \
  --config-name droid_sft_openpi_pytorch_pi05 \
  runner.logger.log_path="${RUN_ROOT}" \
  "$@" 2>&1 | tee "${RUN_LOG_DIR}/train.log"
