#!/usr/bin/env bash

# Start expert-only pi0.5-DROID SFT on the wipe-board LeRobot dataset.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export EMBODIED_PATH="${REPO_ROOT}/examples/sft"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ROOT="${SFT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/pi05_droid_wipe_board_sft}"
RUN_LOG_DIR="${RUN_ROOT}/logs/$(date -u +'%Y%m%d-%H%M%S')"
mkdir -p "${RUN_LOG_DIR}"

echo "Dataset:        ${DATASET_PATH}"
echo "Base checkpoint:${BASE_CHECKPOINT_DIR}"
echo "Norm stats:     ${NORM_STATS_DIR}"
echo "CUDA devices:   ${CUDA_VISIBLE_DEVICES}"
echo "Training logs:  ${RUN_LOG_DIR}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES
"${PYTHON_BIN}" examples/sft/train_vla_sft.py \
  --config-path "${REPO_ROOT}/examples/sft/config" \
  --config-name droid_sft_openpi_pi05 \
  runner.logger.log_path="${RUN_ROOT}" \
  "$@" 2>&1 | tee "${RUN_LOG_DIR}/train.log"
