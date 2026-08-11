#!/usr/bin/env bash

# Convert RLinf's consolidated full_weights.pt into an OpenPI deploy directory.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "${SFT_CHECKPOINT:-}" ]]; then
  echo "Set SFT_CHECKPOINT to a global_step, actor, model_state_dict, or .pt path." >&2
  exit 1
fi

OUTPUT_CHECKPOINT_DIR="${OUTPUT_CHECKPOINT_DIR:-${REPO_ROOT}/outputs/pi05_droid_wipe_board_sft/exported}"
REFERENCE_CHECKPOINT_DIR="${REFERENCE_CHECKPOINT_DIR:-${PI05_DROID_RLINF_MODEL_PATH:-/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch_rlinf}}"
EXPORT_NORM_STATS_DIR="${EXPORT_NORM_STATS_DIR:-${PI05_DROID_RLINF_NORM_STATS:-${REFERENCE_CHECKPOINT_DIR}/assets/wipe_board_v1_zed196_force}}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u \
  toolkits/standalone_eval_scripts/openpi/export_pi05_droid_sft_checkpoint.py \
  --checkpoint "${SFT_CHECKPOINT}" \
  --reference-checkpoint "${REFERENCE_CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_CHECKPOINT_DIR}" \
  --norm-stats-dir "${EXPORT_NORM_STATS_DIR}" \
  "$@"
