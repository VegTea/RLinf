#!/usr/bin/env bash
# DEM dem-v1 checkpoints use the model factory, never runner.ckpt_path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || "$1" == --help ]]; then
    echo "Usage: bash starts/eval_dem_stack_cube.sh POLICY.pt [eval launcher options / Hydra overrides]"
    echo "Default: 1 env, seed 42, first episode only, H=16/K=5, Isaac Sim 6."
    exit 0
fi
export DEM_POLICY_PATH="$(realpath "$1")"
shift
[[ -f "$DEM_POLICY_PATH" && -f "${DEM_POLICY_PATH%.pt}.json" ]] || {
    echo "Missing DEM policy or deployment manifest" >&2; exit 2;
}
export RLINF_DEM_PYTHON="${RLINF_DEM_PYTHON:-${ROOT}/.venv-dem/bin/python}"
[[ -x "$RLINF_DEM_PYTHON" ]] || { echo "Missing DEM interpreter" >&2; exit 2; }
bash "$ROOT/starts/eval_isaaclab_checkpoint_videos.sh" \
    --config-name isaaclab_franka_stack_cube_eval_dem_isaacsim6 \
    "$@"
