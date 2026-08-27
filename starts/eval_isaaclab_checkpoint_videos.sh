#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  eval_isaaclab_checkpoint_videos.sh [CHECKPOINT] [CONFIG_NAME] [OUTPUT_DIR] [HYDRA_OVERRIDE ...]

Evaluate one episode per IsaacLab environment. When CHECKPOINT is omitted, use
the selected config's default rollout.model.model_path. When supplied,
CHECKPOINT must be a .pt state-dict file.
Each video contains the external/table camera on the left and wrist camera on
the right. A JSON file is written for every environment with its scenario_id,
policy_success value, and video path.

Example:
  bash starts/eval_isaaclab_checkpoint_videos.sh \
    /path/to/full_weights.pt \
    isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
    /path/to/eval-output \
    env.eval.total_num_envs=32

Use the nearest100 config's default initial model instead:
  bash starts/eval_isaaclab_checkpoint_videos.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${SCRIPT_DIR}")"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
DEFAULT_CONFIG="isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100"
CHECKPOINT=""
CONFIG_NAME="${DEFAULT_CONFIG}"

if [[ "${1:-}" == "-" ]]; then
    shift
elif [[ -n "${1:-}" && -f "${1}" ]]; then
    CHECKPOINT="$1"
    shift
elif [[ -n "${1:-}" && -f "${EMBODIED_PATH}/config/${1}.yaml" ]]; then
    CONFIG_NAME="$1"
    shift
elif [[ -n "${1:-}" && "${1}" == *=* ]]; then
    :
elif [[ -n "${1:-}" ]]; then
    echo "Expected a checkpoint file, '-' for the config default, or a config name; got: $1" >&2
    exit 2
fi

if [[ -n "${1:-}" && -f "${EMBODIED_PATH}/config/${1}.yaml" ]]; then
    CONFIG_NAME="$1"
    shift
fi

OUTPUT_DIR="${REPO_PATH}/logs/isaaclab-checkpoint-eval-$(date +'%Y%m%d-%H%M%S')"
if [[ -n "${1:-}" && "${1}" != *=* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

if [[ -n "${CHECKPOINT}" && ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint file not found: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml" ]]; then
    echo "Config file not found: ${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
if [[ -n "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
    CHECKPOINT_ARGS=("runner.ckpt_path=${CHECKPOINT}")
    echo "Checkpoint: ${CHECKPOINT}"
else
    CHECKPOINT_ARGS=()
    echo "Checkpoint: config default rollout.model.model_path"
fi
EXTRA_ARGS=("$@")

echo "Config:     ${CONFIG_NAME}"
echo "Output:     ${OUTPUT_DIR}"

LOG_DIR="${OUTPUT_DIR}" bash "${EMBODIED_PATH}/eval_embodiment.sh" \
    "${CONFIG_NAME}" \
    "${ROBOT_PLATFORM:-LIBERO}" \
    "${CHECKPOINT_ARGS[@]}" \
    "runner.resume_dir=null" \
    "algorithm.eval_rollout_epoch=1" \
    "env.eval.video_cfg.save_video=true" \
    "env.eval.video_cfg.info_on_video=true" \
    "++env.eval.video_cfg.per_env_videos=true" \
    "++env.eval.video_cfg.wait_for_video_writes=true" \
    "++env.eval.video_cfg.camera_keys=[main_images,wrist_images]" \
    "++env.eval.per_episode_result_dir=${OUTPUT_DIR}/per_env_results" \
    "++env.eval.per_episode_skip_existing=false" \
    "++env.eval.per_episode_result_file_naming=env_episode" \
    "++runner.eval_manifest_path=${OUTPUT_DIR}/eval_results.json" \
    "${EXTRA_ARGS[@]}"

echo "Videos: ${OUTPUT_DIR}/video/eval/seed_*/*_env_*.mp4"
echo "Per-env JSON: ${OUTPUT_DIR}/per_env_results/*.json"
echo "Combined JSON: ${OUTPUT_DIR}/eval_results.json"
