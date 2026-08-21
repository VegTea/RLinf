#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash examples/embodiment/replay_isaaclab_trajectory_batch.sh TRAJ_ROOT [OUT_DIR] [extra replay args...]

Arguments:
  TRAJ_ROOT  Directory containing success/*.npz and/or fail/*.npz trajectories.
  OUT_DIR    Output directory. Defaults to <TRAJ_ROOT>_replay_<timestamp>.

Environment variables:
  PYTHON_BIN           Python executable. Defaults to <repo>/.venv/bin/python.
  NUM_VISUAL_VARIANTS  Default --num-visual-variants. Defaults to 1.
  REPLAY_MODE          Default --mode. Defaults to state.
  REPLAY_FPS           Optional default --fps.
  REPLAY_CAMERA        Optional default --camera, table or wrist.
  REPLAY_CONFIG        Optional replay yaml path.
  VISUAL_SCENARIO_FILE Default --visual-scenario-file. Defaults to the 10025 combined scenario jsonl.
  VISUAL_SAMPLES_PER_TRAJ
                       Default --visual-samples-per-traj. Defaults to 1.
  VISUAL_SCENARIO_SEED Optional default --visual-scenario-seed.
  SAVE_H5              Set to 1 to pass --save-h5.
  H5_OUTPUT_DIR        Optional --h5-output-dir.
  H5_NUM_VIEWS         Optional --h5-num-views.
  H5_NUM_EXTERNAL_VIEWS
                       Optional --h5-num-external-views.
  H5_VIEW_SCENARIO_FILE
                       Optional --h5-view-scenario-file.
  H5_LIGHT_SCENARIO_FILE
                       Optional --h5-light-scenario-file.
  H5_LIGHT_SELECTION   Optional --h5-light-selection, random or sequential.
  H5_LIGHT_SEED        Optional --h5-light-seed.
  H5_FIXED_LIGHT_ID    Optional --h5-fixed-light-id.
  H5_CUBE_COLOR_MAP_JSON
                       Optional explicit red/blue/green cube map JSON.
  SAVE_LEROBOT         Set to 1 to pass --save-lerobot.
  LEROBOT_OUTPUT_DIR   Required by --save-lerobot.
  LEROBOT_REPO_ID      Optional --lerobot-repo-id.
  LEROBOT_FPS          Optional --lerobot-fps.
  LEROBOT_OVERWRITE    Set to 1 to pass --lerobot-overwrite.

Examples:
  bash examples/embodiment/replay_isaaclab_trajectory_batch.sh \
    /path/to/isaaclab_2k_traj \
    /path/to/isaaclab_2k_traj_replay

  bash examples/embodiment/replay_isaaclab_trajectory_batch.sh \
    /path/to/isaaclab_2k_traj \
    /path/to/isaaclab_2k_traj_replay_quick \
    --max-frames 80
EOF
}

has_arg() {
  local target="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

TRAJ_ROOT="$(realpath "$1")"
shift

if [[ ! -d "$TRAJ_ROOT" ]]; then
  echo "Trajectory root does not exist: $TRAJ_ROOT" >&2
  exit 1
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  OUT_DIR="$(realpath -m "$1")"
  shift
else
  RUN_TAG="$(date +'%Y%m%d_%H%M%S')"
  OUT_DIR="$(realpath -m "${TRAJ_ROOT%/}_replay_${RUN_TAG}")"
fi

EXTRA_ARGS=("$@")
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEFAULT_VISUAL_SCENARIO_FILE="${REPO_ROOT}/rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

SUCCESS_COUNT=0
FAIL_COUNT=0
if [[ -d "${TRAJ_ROOT}/success" ]]; then
  SUCCESS_COUNT="$(find "${TRAJ_ROOT}/success" -type f -name '*.npz' | wc -l)"
fi
if [[ -d "${TRAJ_ROOT}/fail" ]]; then
  FAIL_COUNT="$(find "${TRAJ_ROOT}/fail" -type f -name '*.npz' | wc -l)"
fi
TOTAL_COUNT=$((SUCCESS_COUNT + FAIL_COUNT))

if [[ "$TOTAL_COUNT" -le 0 ]]; then
  echo "No .npz trajectories found under success/ or fail/: $TRAJ_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/batch_replay.log"

REPLAY_ARGS=(
  "${SCRIPT_DIR}/replay_isaaclab_trajectory.py"
  "$TRAJ_ROOT"
  --output-dir "$OUT_DIR"
  --preserve-label-dirs
)

if ! has_arg "--mode" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--mode "${REPLAY_MODE:-state}")
fi
if ! has_arg "--max-files" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--max-files "$TOTAL_COUNT")
fi
if ! has_arg "--num-visual-variants" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--num-visual-variants "${NUM_VISUAL_VARIANTS:-1}")
fi
if [[ -n "${REPLAY_FPS:-}" ]] && ! has_arg "--fps" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--fps "$REPLAY_FPS")
fi
if [[ -n "${REPLAY_CAMERA:-}" ]] && ! has_arg "--camera" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--camera "$REPLAY_CAMERA")
fi
if [[ -n "${REPLAY_CONFIG:-}" ]] && ! has_arg "--config" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--config "$REPLAY_CONFIG")
fi
if ! has_arg "--visual-scenario-file" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--visual-scenario-file "${VISUAL_SCENARIO_FILE:-$DEFAULT_VISUAL_SCENARIO_FILE}")
fi
if ! has_arg "--visual-samples-per-traj" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--visual-samples-per-traj "${VISUAL_SAMPLES_PER_TRAJ:-1}")
fi
if [[ -n "${VISUAL_SCENARIO_SEED:-}" ]] && ! has_arg "--visual-scenario-seed" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--visual-scenario-seed "$VISUAL_SCENARIO_SEED")
fi
if [[ "${SAVE_H5:-0}" == "1" ]] && ! has_arg "--save-h5" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--save-h5)
fi
if [[ -n "${H5_OUTPUT_DIR:-}" ]] && ! has_arg "--h5-output-dir" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-output-dir "$H5_OUTPUT_DIR")
fi
if [[ -n "${H5_NUM_VIEWS:-}" ]] && ! has_arg "--h5-num-views" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-num-views "$H5_NUM_VIEWS")
fi
if [[ -n "${H5_NUM_EXTERNAL_VIEWS:-}" ]] && ! has_arg "--h5-num-external-views" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-num-external-views "$H5_NUM_EXTERNAL_VIEWS")
fi
if [[ -n "${H5_VIEW_SCENARIO_FILE:-}" ]] && ! has_arg "--h5-view-scenario-file" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-view-scenario-file "$H5_VIEW_SCENARIO_FILE")
fi
if [[ -n "${H5_LIGHT_SCENARIO_FILE:-}" ]] && ! has_arg "--h5-light-scenario-file" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-light-scenario-file "$H5_LIGHT_SCENARIO_FILE")
fi
if [[ -n "${H5_LIGHT_SELECTION:-}" ]] && ! has_arg "--h5-light-selection" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-light-selection "$H5_LIGHT_SELECTION")
fi
if [[ -n "${H5_LIGHT_SEED:-}" ]] && ! has_arg "--h5-light-seed" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-light-seed "$H5_LIGHT_SEED")
fi
if [[ -n "${H5_FIXED_LIGHT_ID:-}" ]] && ! has_arg "--h5-fixed-light-id" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-fixed-light-id "$H5_FIXED_LIGHT_ID")
fi
if [[ -n "${H5_CUBE_COLOR_MAP_JSON:-}" ]] && ! has_arg "--h5-cube-color-map-json" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--h5-cube-color-map-json "$H5_CUBE_COLOR_MAP_JSON")
fi
if [[ "${SAVE_LEROBOT:-0}" == "1" ]] && ! has_arg "--save-lerobot" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--save-lerobot)
fi
if [[ -n "${LEROBOT_OUTPUT_DIR:-}" ]] && ! has_arg "--lerobot-output-dir" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--lerobot-output-dir "$LEROBOT_OUTPUT_DIR")
fi
if [[ -n "${LEROBOT_REPO_ID:-}" ]] && ! has_arg "--lerobot-repo-id" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--lerobot-repo-id "$LEROBOT_REPO_ID")
fi
if [[ -n "${LEROBOT_FPS:-}" ]] && ! has_arg "--lerobot-fps" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--lerobot-fps "$LEROBOT_FPS")
fi
if [[ "${LEROBOT_OVERWRITE:-0}" == "1" ]] && ! has_arg "--lerobot-overwrite" "${EXTRA_ARGS[@]}"; then
  REPLAY_ARGS+=(--lerobot-overwrite)
fi

REPLAY_ARGS+=("${EXTRA_ARGS[@]}")

{
  echo "[batch_replay] repo: $REPO_ROOT"
  echo "[batch_replay] trajectory root: $TRAJ_ROOT"
  echo "[batch_replay] output dir: $OUT_DIR"
  echo "[batch_replay] success trajectories: $SUCCESS_COUNT"
  echo "[batch_replay] fail trajectories: $FAIL_COUNT"
  echo "[batch_replay] total trajectories: $TOTAL_COUNT"
  echo "[batch_replay] python: $PYTHON_BIN"
  printf '[batch_replay] command:'
  printf ' %q' "$PYTHON_BIN" "${REPLAY_ARGS[@]}"
  printf '\n'
} | tee "$LOG_FILE"

cd "$REPO_ROOT"
"$PYTHON_BIN" "${REPLAY_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[batch_replay] done: $OUT_DIR" | tee -a "$LOG_FILE"
