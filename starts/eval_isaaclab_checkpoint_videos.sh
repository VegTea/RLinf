#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  eval_isaaclab_checkpoint_videos.sh [OPTIONS] [CHECKPOINT] [CONFIG_NAME] [OUTPUT_DIR] [HYDRA_OVERRIDE ...]

Options:
  --checkpoint PATH     Load a .pt state-dict checkpoint.
  --config-name NAME    Select an examples/embodiment/config YAML.
  --output-dir PATH     Write all evaluation artifacts under PATH.
  --scenario-file PATH  Load scenarios from this JSONL file. The selected
                        config still controls scenario_reset.mode and loop.
  -h, --help            Show this help.

If --scenario-file is omitted, the selected YAML controls reset behavior. A
config with scenario_reset disabled uses IsaacLab's native random reset; a
config with it enabled uses its configured scenario file. In both cases the
actual post-reset settings are saved to OUTPUT_DIR/eval_scenarios.jsonl.
Success throughput is saved to OUTPUT_DIR/eval_throughput.json using first
success steps for successful envs and full episode lengths for failed envs.

Legacy positional arguments remain supported. Use '-' as CHECKPOINT to retain
the config's default rollout.model.model_path.

Examples:
  bash starts/eval_isaaclab_checkpoint_videos.sh \
    --scenario-file /path/to/scenarios.jsonl \
    --config-name isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
    env.eval.total_num_envs=32

  bash starts/eval_isaaclab_checkpoint_videos.sh \
    --checkpoint /path/to/full_weights.pt \
    --config-name isaaclab_franka_stack_cube_ppo_openpi_pi05 \
    /path/to/eval-output \
    env.eval.total_num_envs=32
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${SCRIPT_DIR}")"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
DEFAULT_CONFIG="isaaclab_franka_stack_cube_ppo_openpi_pi05"
CALLER_DIR="$(pwd)"

CHECKPOINT=""
CONFIG_NAME=""
OUTPUT_DIR=""
SCENARIO_FILE=""
CHECKPOINT_SET=false
CONFIG_SET=false
OUTPUT_SET=false
POSITIONAL=()
EXTRA_ARGS=()

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "--checkpoint requires a path" >&2; exit 2; }
            CHECKPOINT="$2"
            CHECKPOINT_SET=true
            shift 2
            ;;
        --config-name)
            [[ $# -ge 2 ]] || { echo "--config-name requires a name" >&2; exit 2; }
            CONFIG_NAME="$2"
            CONFIG_SET=true
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "--output-dir requires a path" >&2; exit 2; }
            OUTPUT_DIR="$2"
            OUTPUT_SET=true
            shift 2
            ;;
        --scenario-file)
            [[ $# -ge 2 ]] || { echo "--scenario-file requires a path" >&2; exit 2; }
            SCENARIO_FILE="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *=*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

pos_index=0
if [[ "${CHECKPOINT_SET}" == false && ${#POSITIONAL[@]} -gt ${pos_index} ]]; then
    candidate="${POSITIONAL[${pos_index}]}"
    if [[ "${candidate}" == "-" ]]; then
        ((pos_index += 1))
    elif [[ -f "${candidate}" ]]; then
        CHECKPOINT="${candidate}"
        ((pos_index += 1))
    elif [[ -f "${EMBODIED_PATH}/config/${candidate}.yaml" ]]; then
        CONFIG_NAME="${candidate}"
        CONFIG_SET=true
        ((pos_index += 1))
    else
        echo "Expected a checkpoint file, '-', or a config name; got: ${candidate}" >&2
        exit 2
    fi
fi
if [[ "${CONFIG_SET}" == false && ${#POSITIONAL[@]} -gt ${pos_index} ]]; then
    CONFIG_NAME="${POSITIONAL[${pos_index}]}"
    ((pos_index += 1))
fi
if [[ "${OUTPUT_SET}" == false && ${#POSITIONAL[@]} -gt ${pos_index} ]]; then
    OUTPUT_DIR="${POSITIONAL[${pos_index}]}"
    ((pos_index += 1))
fi
if [[ ${#POSITIONAL[@]} -gt ${pos_index} ]]; then
    echo "Unexpected positional argument: ${POSITIONAL[${pos_index}]}" >&2
    exit 2
fi

CONFIG_NAME="${CONFIG_NAME:-${DEFAULT_CONFIG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_PATH}/logs/isaaclab-checkpoint-eval-$(date +'%Y%m%d-%H%M%S')}"

if [[ -n "${CHECKPOINT}" && ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint file not found: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -f "${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml" ]]; then
    echo "Config file not found: ${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml" >&2
    exit 2
fi
if [[ -n "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
fi
if [[ -n "${SCENARIO_FILE}" ]]; then
    if [[ ! -f "${SCENARIO_FILE}" ]]; then
        echo "Scenario file not found: ${SCENARIO_FILE}" >&2
        exit 2
    fi
    SCENARIO_FILE="$(cd "$(dirname "${SCENARIO_FILE}")" && pwd)/$(basename "${SCENARIO_FILE}")"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${CALLER_DIR}/${OUTPUT_DIR}"
fi

source "${SCRIPT_DIR}/source.sh"

SCENARIO_ARGS=()
if [[ -n "${SCENARIO_FILE}" ]]; then
    python -m rlinf.envs.isaaclab.scenario_export --validate "${SCENARIO_FILE}"
    SCENARIO_ARGS=(
        "env.eval.init_params.scenario_reset.enabled=true"
        "env.eval.init_params.scenario_reset.scenario_file=${SCENARIO_FILE}"
    )
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=("runner.ckpt_path=${CHECKPOINT}")
    echo "Checkpoint: ${CHECKPOINT}"
else
    echo "Checkpoint: config default rollout.model.model_path"
fi

echo "Config:     ${CONFIG_NAME}"
echo "Output:     ${OUTPUT_DIR}"
if [[ -n "${SCENARIO_FILE}" ]]; then
    echo "Scenarios:  ${SCENARIO_FILE} (mode/loop from config)"
else
    echo "Scenarios:  selected config reset behavior"
fi

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
    "++runner.eval_scenario_output_path=${OUTPUT_DIR}/eval_scenarios.jsonl" \
    "++runner.eval_throughput_path=${OUTPUT_DIR}/eval_throughput.json" \
    "${EXTRA_ARGS[@]}" \
    "${SCENARIO_ARGS[@]}"

echo "Videos: ${OUTPUT_DIR}/video/eval/seed_*/*_env_*.mp4"
echo "Per-env JSON: ${OUTPUT_DIR}/per_env_results/*.json"
echo "Results: ${OUTPUT_DIR}/eval_results.json"
echo "Success throughput: ${OUTPUT_DIR}/eval_throughput.json"
echo "Reloadable scenarios: ${OUTPUT_DIR}/eval_scenarios.jsonl"
