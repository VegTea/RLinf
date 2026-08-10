#!/usr/bin/env bash

# Validate one four-H100 update followed by deterministic decoded evaluation.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_TIMESTAMP="$(date -u +'%Y%m%d-%H%M%S')"
SMOKE_ROOT="${H100_SMOKE_ROOT:-${REPO_ROOT}/outputs/pi05_droid_h100_smoke/${RUN_TIMESTAMP}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
MAX_STEPS=1 \
SAVE_INTERVAL=-1 \
VAL_CHECK_INTERVAL=1 \
EVAL_CHUNKS_PER_EPISODE=1 \
H100_TRAIN_ROOT="${SMOKE_ROOT}" \
bash "${SCRIPT_DIR}/run_h100_train.sh" \
  runner.logger.experiment_name=pi05_droid_wipe_board_h100_smoke

echo "Smoke test complete: ${SMOKE_ROOT}"
