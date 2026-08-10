#!/usr/bin/env bash

# Compile, update once, and decode one held-out chunk per episode on 4xH100.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_ROOT="${H100_SMOKE_ROOT:-${SCRIPT_DIR}/../../../outputs/pi05_droid_openpi_pytorch_h100_smoke/$(date -u +'%Y%m%d-%H%M%S')}"

MAX_STEPS=1 \
SAVE_INTERVAL=-1 \
VAL_CHECK_INTERVAL=1 \
EVAL_CHUNKS_PER_EPISODE=1 \
H100_TRAIN_ROOT="${SMOKE_ROOT}" \
bash "${SCRIPT_DIR}/run_openpi_pytorch_h100_train.sh" \
  runner.logger.experiment_name=pi05_droid_openpi_pytorch_h100_smoke
