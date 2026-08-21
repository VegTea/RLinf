#!/usr/bin/env bash

set -euo pipefail

export PYTHONWARNINGS="ignore::FutureWarning"

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

CONFIG_NAME="isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum"
GPU_COUNT="1"
TRAIN_ENVS=""
EVAL_ENVS="1"
MAX_EPISODE_STEPS="450"
ROLLOUT_EPOCH="1"
UPDATE_EPOCH="1"
ACTOR_GLOBAL_BATCH_SIZE=""
ACTOR_MICRO_BATCH_SIZE="32"
LOG_ROOT="${REPO_PATH}/logs/gpu_smoke"
LOG_DIR_OVERRIDE=""
SAMPLE_INTERVAL_SECONDS="5"
EXTRA_OVERRIDES=()

usage() {
    cat <<'EOF'
Usage:
  bash examples/embodiment/check_isaaclab_gpu_smoke.sh [options] [-- Hydra overrides...]

Purpose:
  Smoke test IsaacLab GPU health by reusing the existing training pipeline:
  launch IsaacLab envs, set JSONL scenarios, load the OpenPI model, generate
  actions through rollout workers, step the envs, and run one tiny actor update.

Options:
  --gpus N                Number of visible GPUs to use. Default: 1.
  --config NAME           Hydra config name. Default:
                          isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum
  --train-envs N          Total train envs. Defaults to 16 per visible GPU.
  --eval-envs N           Total eval envs kept in config. Default: 1.
  --max-episode-steps N   Env rollout length. Default: 450.
  --rollout-epoch N       Rollout epochs for this smoke run. Default: 1.
  --update-epoch N        Actor update epochs for this smoke run. Default: 1.
  --actor-global-batch N  Actor global batch size. Defaults to 288 per visible
                          GPU.
  --actor-micro-batch N   Actor micro batch size. Default: 32.
  --log-root DIR          Parent log directory. Default: RLinf/logs/gpu_smoke.
  --log-dir DIR           Exact log directory for this run. Mutually exclusive
                          with the timestamped directory under --log-root.
  --sample-interval SEC   nvidia-smi sampling interval. Set 0 to disable.
  -h, --help              Show this help.

Examples:
  # Single-GPU smoke test on the first visible GPU
  bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 1

  # Two-GPU smoke test
  bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 2

  # Eight-GPU smoke test, similar env pressure to training
  bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 8

  # Pin single-GPU test to physical GPU 3
  CUDA_VISIBLE_DEVICES=3 bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 1

  # Add arbitrary Hydra overrides after --
  bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 8 -- actor.optim.lr=1e-6
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_COUNT="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --train-envs)
            TRAIN_ENVS="$2"
            shift 2
            ;;
        --eval-envs)
            EVAL_ENVS="$2"
            shift 2
            ;;
        --max-episode-steps)
            MAX_EPISODE_STEPS="$2"
            shift 2
            ;;
        --rollout-epoch)
            ROLLOUT_EPOCH="$2"
            shift 2
            ;;
        --update-epoch)
            UPDATE_EPOCH="$2"
            shift 2
            ;;
        --actor-global-batch)
            ACTOR_GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --actor-micro-batch)
            ACTOR_MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --sample-interval)
            SAMPLE_INTERVAL_SECONDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_OVERRIDES+=("$@")
            break
            ;;
        *)
            EXTRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --gpus must be a positive integer, got '${GPU_COUNT}'." >&2
    exit 2
fi

if ! [[ "${SAMPLE_INTERVAL_SECONDS}" =~ ^([0-9]+|[0-9]+\.[0-9]+)$ ]]; then
    echo "ERROR: --sample-interval must be a non-negative number, got '${SAMPLE_INTERVAL_SECONDS}'." >&2
    exit 2
fi

if [[ -n "${LOG_DIR_OVERRIDE}" && "${LOG_DIR_OVERRIDE}" == "${LOG_ROOT}"/* ]]; then
    echo "ERROR: use either --log-dir or --log-root; --log-dir must not be nested under --log-root." >&2
    exit 2
fi

if [[ -z "${TRAIN_ENVS}" ]]; then
    TRAIN_ENVS="$((GPU_COUNT * 16))"
fi

if [[ -z "${ACTOR_GLOBAL_BATCH_SIZE}" ]]; then
    # Keep the default per-rank batch at 288. With the default 16 envs/GPU,
    # 450 steps, and five-action chunks, this divides the 1440 samples/rank.
    ACTOR_GLOBAL_BATCH_SIZE="$((GPU_COUNT * 288))"
fi

export NVIDIA_DRIVER_CAPABILITIES=all
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
export ISAAC_PATH="${ISAAC_PATH:-${REPO_PATH}/isaac_sim}"
export EXP_PATH="${EXP_PATH:-${ISAAC_PATH}/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_PATH}/kit}"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export EMBODIED_PATH

if [[ -f "${ISAAC_PATH}/setup_python_env.sh" ]]; then
    # Isaac Sim's Python modules (including `isaacsim`) are exposed by this
    # setup script. Keep the standalone smoke launcher consistent with the
    # regular run_embodiment.sh entry point. The vendor script reads optional
    # variables without defaults, so temporarily disable nounset while sourcing.
    set +u
    source "${ISAAC_PATH}/setup_python_env.sh"
    set -u
else
    echo "ERROR: Isaac Sim environment setup not found: ${ISAAC_PATH}/setup_python_env.sh" >&2
    exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((GPU_COUNT - 1))")
    export CUDA_VISIBLE_DEVICES
fi

if [[ -x "${REPO_PATH}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_PATH}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

if [[ -n "${LOG_DIR_OVERRIDE}" ]]; then
    LOG_DIR="${LOG_DIR_OVERRIDE}"
else
    LOG_DIR="${LOG_ROOT}/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-${GPU_COUNT}gpu"
fi
MEGA_LOG_FILE="${LOG_DIR}/gpu_smoke.log"
GPU_USAGE_FILE="${LOG_DIR}/gpu_usage.csv"
GPU_PEAK_FILE="${LOG_DIR}/gpu_peak.csv"
SAMPLER_PID=""

if [[ -e "${LOG_DIR}" ]]; then
    echo "ERROR: log directory already exists: ${LOG_DIR}" >&2
    exit 2
fi
mkdir -p "${LOG_DIR}"

sample_gpu_usage() {
    nvidia-smi \
        --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F ', ' '{print $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8}' \
        >> "${GPU_USAGE_FILE}" || true
}

stop_gpu_sampler() {
    if [[ -n "${SAMPLER_PID}" ]]; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
        SAMPLER_PID=""
    fi
}

write_gpu_peak_summary() {
    if [[ ! -s "${GPU_USAGE_FILE}" ]]; then
        return
    fi

    awk -F ',' '
        NR == 1 { next }
        {
            gpu = $2
            used = $5 + 0
            util = $7 + 0
            if (!(gpu in max_used) || used > max_used[gpu]) max_used[gpu] = used
            if (!(gpu in max_util) || util > max_util[gpu]) max_util[gpu] = util
            total[gpu] = $6
        }
        END {
            print "gpu_index,peak_memory_mib,total_memory_mib,peak_utilization_percent"
            for (gpu in max_used) {
                print gpu "," max_used[gpu] "," total[gpu] "," max_util[gpu]
            }
        }
    ' "${GPU_USAGE_FILE}" | sort -t, -k1,1n > "${GPU_PEAK_FILE}"
}

trap stop_gpu_sampler EXIT

OVERRIDES=(
    "runner.logger.log_path=${LOG_DIR}"
    "runner.max_steps=1"
    "runner.max_epochs=1"
    "runner.save_interval=1000000"
    "runner.val_check_interval=-1"
    "runner.only_eval=false"
    "env.train.total_num_envs=${TRAIN_ENVS}"
    "env.train.max_episode_steps=${MAX_EPISODE_STEPS}"
    "env.train.max_steps_per_rollout_epoch=${MAX_EPISODE_STEPS}"
    "env.eval.total_num_envs=${EVAL_ENVS}"
    "env.eval.video_cfg.save_video=false"
    "env.eval.max_episode_steps=${MAX_EPISODE_STEPS}"
    "env.eval.max_steps_per_rollout_epoch=${MAX_EPISODE_STEPS}"
    "algorithm.rollout_epoch=${ROLLOUT_EPOCH}"
    "algorithm.eval_rollout_epoch=1"
    "algorithm.update_epoch=${UPDATE_EPOCH}"
    "actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}"
    "actor.micro_batch_size=${ACTOR_MICRO_BATCH_SIZE}"
)

OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

CMD=(
    "${PYTHON_BIN}" "${SRC_FILE}"
    "--config-path" "${EMBODIED_PATH}/config/"
    "--config-name" "${CONFIG_NAME}"
    "${OVERRIDES[@]}"
)

{
    echo "===== RLinf IsaacLab GPU smoke test ====="
    echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "hostname=$(hostname)"
    echo "pwd=$(pwd)"
    echo "config_name=${CONFIG_NAME}"
    echo "config_path=${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
    echo "gpu_count_requested=${GPU_COUNT}"
    echo "train_envs=${TRAIN_ENVS}"
    echo "eval_envs=${EVAL_ENVS}"
    echo "max_episode_steps=${MAX_EPISODE_STEPS}"
    echo "rollout_epoch=${ROLLOUT_EPOCH}"
    echo "update_epoch=${UPDATE_EPOCH}"
    echo "actor_global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}"
    echo "actor_micro_batch_size=${ACTOR_MICRO_BATCH_SIZE}"
    echo "gpu_sample_interval_seconds=${SAMPLE_INTERVAL_SECONDS}"
    echo "gpu_usage_file=${GPU_USAGE_FILE}"
    echo "robot_platform=${ROBOT_PLATFORM}"
    echo "libero_type=${LIBERO_TYPE}"
    echo "python=${PYTHON_BIN}"
    echo "repo_path=${REPO_PATH}"
    echo "log_dir=${LOG_DIR}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "nvidia_visible_devices=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia_smi_gpu_count=$(nvidia-smi -L | wc -l)"
        nvidia-smi -L | sed 's/^/gpu: /'
        echo "----- nvidia-smi before -----"
        nvidia-smi
    else
        echo "nvidia_smi_gpu_count=<nvidia-smi not found>"
    fi
    printf 'cmd='
    printf '%q ' "${CMD[@]}"
    printf '\n'
    echo "===== end smoke test header ====="
} > "${MEGA_LOG_FILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    visible_gpu_count=$(nvidia-smi -L | wc -l)
    if (( visible_gpu_count < GPU_COUNT )); then
        echo "ERROR: requested ${GPU_COUNT} GPUs but only ${visible_gpu_count} are visible." | tee -a "${MEGA_LOG_FILE}"
        exit 2
    fi
    echo "timestamp,gpu_index,gpu_uuid,gpu_name,memory_used_mib,memory_total_mib,utilization_gpu_percent,utilization_memory_percent" > "${GPU_USAGE_FILE}"
    if awk "BEGIN { exit !(${SAMPLE_INTERVAL_SECONDS} > 0) }"; then
        sample_gpu_usage
        (
            while true; do
                sleep "${SAMPLE_INTERVAL_SECONDS}"
                sample_gpu_usage
            done
        ) &
        SAMPLER_PID=$!
    fi
fi

set +e
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e
stop_gpu_sampler
write_gpu_peak_summary

{
    echo "===== smoke test result ====="
    echo "exit_status=${STATUS}"
    echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "----- nvidia-smi after -----"
        nvidia-smi
        if [[ -s "${GPU_PEAK_FILE}" ]]; then
            echo "----- GPU peak summary -----"
            cat "${GPU_PEAK_FILE}"
        fi
    fi
    if [[ "${STATUS}" -eq 0 ]]; then
        echo "SMOKE_TEST_PASS: env launch, scenario reset, model load, action rollout, env interaction, and one actor update completed."
    else
        echo "SMOKE_TEST_FAIL: inspect this log and worker_logs under ${LOG_DIR}."
    fi
} | tee -a "${MEGA_LOG_FILE}"

exit "${STATUS}"
