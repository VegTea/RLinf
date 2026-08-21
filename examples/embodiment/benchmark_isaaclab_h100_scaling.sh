#!/usr/bin/env bash

# Run a reproducible single-node GPU scaling study using the existing IsaacLab
# training pipeline. It intentionally leaves the production YAML unchanged.
set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
SMOKE_SCRIPT="${EMBODIED_PATH}/check_isaaclab_gpu_smoke.sh"

CONFIG_NAME="isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100"
GPU_COUNT="2"
ENV_COUNTS="64 128 192 256"
MICRO_BATCHES="4 8 16 32"
GLOBAL_BATCH_SIZE="576"
ROLLOUT_STEPS="450"
LOG_ROOT="${REPO_PATH}/logs/h100_scaling"
PHASE="all"
SELECTED_ENV=""
REPRESENTATIVE_STEPS="2"

usage() {
    cat <<'EOF'
Usage:
  bash examples/embodiment/benchmark_isaaclab_h100_scaling.sh [options]

Runs total-environment and micro-batch sweeps with one rollout/update per
point, then reruns the two selected candidates for multiple training steps.

Options:
  --gpus N                 Visible GPUs in this one-node study. Default: 2.
  --envs "N ..."           Total train env counts. Default: "64 128 192 256".
  --micro-batches "N ..."  Per-rank micro batch sizes. Default: "4 8 16 32".
  --global-batch N         Global PPO batch. Default: 576 (288 per rank on 2 GPUs).
  --rollout-steps N        Steps per environment per rollout. Default: 450.
  --phase env|micro|representative|all
  --selected-env N         Required for --phase micro or representative.
  --representative-steps N Training steps for final candidate checks. Default: 2.
  --log-root DIR           Output root. Default: RLinf/logs/h100_scaling.
  -h, --help               Show this help.

The global batch must be divisible by GPU_COUNT * micro_batch_size. Each
environment count should be divisible by GPU_COUNT. The script records a CSV
with peak GPU memory and derived environment steps/s for every run.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPU_COUNT="$2"; shift 2 ;;
        --envs) ENV_COUNTS="$2"; shift 2 ;;
        --micro-batches) MICRO_BATCHES="$2"; shift 2 ;;
        --global-batch) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --rollout-steps) ROLLOUT_STEPS="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --selected-env) SELECTED_ENV="$2"; shift 2 ;;
        --representative-steps) REPRESENTATIVE_STEPS="$2"; shift 2 ;;
        --log-root) LOG_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'." >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${ROLLOUT_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GPU, global batch, and rollout-step values must be positive integers." >&2
    exit 2
fi
if [[ "${PHASE}" != "env" && "${PHASE}" != "micro" && "${PHASE}" != "representative" && "${PHASE}" != "all" ]]; then
    echo "ERROR: --phase must be env, micro, representative, or all." >&2
    exit 2
fi
if [[ ( "${PHASE}" == "micro" || "${PHASE}" == "representative" ) && -z "${SELECTED_ENV}" ]]; then
    echo "ERROR: --selected-env is required for --phase ${PHASE}." >&2
    exit 2
fi

RUN_ROOT="${LOG_ROOT}/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-${GPU_COUNT}gpu"
RESULTS_CSV="${RUN_ROOT}/benchmark_results.csv"
mkdir -p "${RUN_ROOT}"
printf '%s\n' 'phase,envs,micro_batch_size,global_batch_size,rollout_steps,training_steps,status,step_seconds,env_steps_per_second,peak_memory_mib,log_dir' > "${RESULTS_CSV}"

validate_case() {
    local envs="$1"
    local micro_batch="$2"
    if (( envs % GPU_COUNT != 0 )); then
        echo "ERROR: env count ${envs} is not divisible by ${GPU_COUNT} GPUs." >&2
        return 1
    fi
    if (( GLOBAL_BATCH_SIZE % (GPU_COUNT * micro_batch) != 0 )); then
        echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} is not divisible by ${GPU_COUNT} * ${micro_batch}." >&2
        return 1
    fi
}

record_case() {
    local phase="$1"
    local envs="$2"
    local micro_batch="$3"
    local training_steps="$4"
    local label="$5"
    local run_dir="${RUN_ROOT}/${label}"
    local status="failed"
    local step_seconds=""
    local env_steps_per_second=""
    local peak_memory_mib=""

    validate_case "${envs}" "${micro_batch}"
    echo "===== benchmark ${label} =====" | tee -a "${RUN_ROOT}/benchmark_driver.log"
    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        bash "${SMOKE_SCRIPT}" \
        --gpus "${GPU_COUNT}" \
        --config "${CONFIG_NAME}" \
        --train-envs "${envs}" \
        --eval-envs 1 \
        --max-episode-steps "${ROLLOUT_STEPS}" \
        --rollout-epoch 1 \
        --update-epoch 1 \
        --actor-global-batch "${GLOBAL_BATCH_SIZE}" \
        --actor-micro-batch "${micro_batch}" \
        --log-dir "${run_dir}" \
        --sample-interval 5 \
        -- "runner.max_steps=${training_steps}" "runner.max_epochs=${training_steps}" \
        "runner.save_interval=1000000" \
        |& tee -a "${RUN_ROOT}/benchmark_driver.log"
    command_status=${PIPESTATUS[0]}
    set -e

    # The smoke launcher can return a wrapper/cleanup status different from
    # the training process even after it records an explicit zero exit status.
    # Treat the log's pass marker as authoritative for benchmark accounting.
    if { command -v rg >/dev/null 2>&1 && rg -q 'SMOKE_TEST_PASS' "${run_dir}/gpu_smoke.log" && rg -q 'exit_status=0' "${run_dir}/gpu_smoke.log"; } || \
       { ! command -v rg >/dev/null 2>&1 && grep -Eq 'SMOKE_TEST_PASS' "${run_dir}/gpu_smoke.log" && grep -Eq 'exit_status=0' "${run_dir}/gpu_smoke.log"; }; then
        status="passed"
        step_seconds=$(sed -nE 's/.*step=([0-9]+(\.[0-9]+)?).*/\1/p' "${run_dir}/gpu_smoke.log" | tail -1)
        if [[ -n "${step_seconds}" ]]; then
            env_steps_per_second=$(awk -v envs="${envs}" -v steps="${ROLLOUT_STEPS}" -v seconds="${step_seconds}" 'BEGIN { printf "%.3f", (envs * steps) / seconds }')
        fi
        if [[ -s "${run_dir}/gpu_peak.csv" ]]; then
            # Isaac's sampler writes a data row, a CSV header, then the second
            # GPU row. Select numeric rows so the header cannot become a value.
            peak_memory_mib=$(awk -F ',' '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $2 > max { max = $2 } END { if (max != "") print max }' "${run_dir}/gpu_peak.csv")
        fi
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${phase}" "${envs}" "${micro_batch}" "${GLOBAL_BATCH_SIZE}" "${ROLLOUT_STEPS}" \
        "${training_steps}" "${status}" "${step_seconds}" "${env_steps_per_second}" \
        "${peak_memory_mib}" "${run_dir}" >> "${RESULTS_CSV}"
}

best_env_from_results() {
    awk -F ',' '
        NR > 1 && $1 == "env" && $7 == "passed" && ($9 + 0) > best {
            best = $9 + 0
            env = $2
        }
        END { print env }
    ' "${RESULTS_CSV}"
}

best_micro_from_results() {
    local envs="$1"
    awk -F ',' -v envs="${envs}" '
        NR > 1 && $1 == "micro" && $2 == envs && $7 == "passed" && ($9 + 0) > best {
            best = $9 + 0
            micro = $3
        }
        END { print micro }
    ' "${RESULTS_CSV}"
}

if [[ "${PHASE}" == "env" || "${PHASE}" == "all" ]]; then
    for envs in ${ENV_COUNTS}; do
        record_case "env" "${envs}" 8 1 "env-${envs}-micro-8"
    done
    if [[ "${PHASE}" == "all" ]]; then
        SELECTED_ENV=$(best_env_from_results)
        if [[ -z "${SELECTED_ENV}" ]]; then
            echo "ERROR: no environment-sweep point passed; see ${RESULTS_CSV}." >&2
            exit 1
        fi
    fi
fi

if [[ "${PHASE}" == "micro" || "${PHASE}" == "all" ]]; then
    for micro_batch in ${MICRO_BATCHES}; do
        record_case "micro" "${SELECTED_ENV}" "${micro_batch}" 1 "micro-${micro_batch}-env-${SELECTED_ENV}"
    done
    if [[ "${PHASE}" == "all" ]]; then
        SELECTED_MICRO=$(best_micro_from_results "${SELECTED_ENV}")
        if [[ -z "${SELECTED_MICRO}" ]]; then
            echo "ERROR: no micro-batch point passed; see ${RESULTS_CSV}." >&2
            exit 1
        fi
    fi
fi

if [[ "${PHASE}" == "representative" ]]; then
    SELECTED_MICRO=8
elif [[ "${PHASE}" == "all" ]]; then
    : "${SELECTED_MICRO:?selected micro batch missing}"
else
    SELECTED_MICRO=""
fi

if [[ "${PHASE}" == "representative" || "${PHASE}" == "all" ]]; then
    record_case "representative_throughput" "${SELECTED_ENV}" "${SELECTED_MICRO}" "${REPRESENTATIVE_STEPS}" "representative-throughput"
    record_case "representative_headroom" "${SELECTED_ENV}" 4 "${REPRESENTATIVE_STEPS}" "representative-headroom"
fi

echo "Benchmark results: ${RESULTS_CSV}"
cat "${RESULTS_CSV}"
