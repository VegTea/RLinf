#!/usr/bin/env bash

# Run the production expert-only pi0.5-DROID SFT recipe on four H100 GPUs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
EVAL_CHUNKS_PER_EPISODE="${EVAL_CHUNKS_PER_EPISODE:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
MIN_LR="${MIN_LR:-2.5e-6}"
RUN_TIMESTAMP="$(date -u +'%Y%m%d-%H%M%S')"
RUN_ROOT="${H100_TRAIN_ROOT:-${REPO_ROOT}/outputs/pi05_droid_h100_train/${RUN_TIMESTAMP}}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate the H100 training node." >&2
  exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Dataset directory not found: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CHECKPOINT_DIR}/model.safetensors" ]]; then
  echo "PyTorch checkpoint not found: ${BASE_CHECKPOINT_DIR}/model.safetensors" >&2
  exit 1
fi
if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "Normalization statistics not found: ${NORM_STATS_DIR}/norm_stats.json" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPU_IDS[@]}"
if (( GPU_COUNT != EXPECTED_GPU_COUNT )); then
  echo "Expected ${EXPECTED_GPU_COUNT} visible GPUs, got ${GPU_COUNT}: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

echo "Validated GPUs:"
for gpu_id in "${GPU_IDS[@]}"; do
  gpu_id="${gpu_id//[[:space:]]/}"
  gpu_name="$(nvidia-smi --id="${gpu_id}" --query-gpu=name --format=csv,noheader | head -n 1)"
  gpu_memory_mib="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
  if [[ "${gpu_name}" != *H100* && "${ALLOW_NON_H100:-0}" != "1" ]]; then
    echo "Expected an NVIDIA H100, found GPU ${gpu_id}: ${gpu_name}" >&2
    echo "Set ALLOW_NON_H100=1 only for dry-run validation on other hardware." >&2
    exit 1
  fi
  echo "  GPU ${gpu_id}: ${gpu_name} (${gpu_memory_mib} MiB)"
done

samples_per_micro_step=$(( MICRO_BATCH_SIZE * GPU_COUNT ))
if (( GLOBAL_BATCH_SIZE < samples_per_micro_step )); then
  echo "global batch ${GLOBAL_BATCH_SIZE} is smaller than one distributed micro step (${samples_per_micro_step})." >&2
  exit 1
fi
if (( GLOBAL_BATCH_SIZE % samples_per_micro_step != 0 )); then
  echo "global batch ${GLOBAL_BATCH_SIZE} must be divisible by micro batch x GPUs (${samples_per_micro_step})." >&2
  exit 1
fi
gradient_accumulation=$(( GLOBAL_BATCH_SIZE / samples_per_micro_step ))
last_gpu_rank=$(( GPU_COUNT - 1 ))

TRAIN_ARGS=(
  "cluster.component_placement.actor=0-${last_gpu_rank}"
  "runner.max_steps=${MAX_STEPS}"
  "runner.save_interval=${SAVE_INTERVAL}"
  "runner.val_check_interval=${VAL_CHECK_INTERVAL}"
  "data.episode_split.eval_chunks_per_episode=${EVAL_CHUNKS_PER_EPISODE}"
  "runner.logger.experiment_name=pi05_droid_wipe_board_h100"
  "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
  "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
  "actor.model.num_steps=10"
  "actor.model.openpi.train_expert_only=true"
  "actor.optim.lr=${LEARNING_RATE}"
  "actor.optim.lr_scheduler=openpi_cosine"
  "actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
  "actor.optim.min_lr=${MIN_LR}"
)
TRAIN_ARGS+=("$@")

echo "Dataset:              ${DATASET_PATH}"
echo "Base checkpoint:      ${BASE_CHECKPOINT_DIR}"
echo "Norm stats:           ${NORM_STATS_DIR}"
echo "Output root:          ${RUN_ROOT}"
echo "Micro/global batch:   ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}"
echo "Gradient accumulation:${gradient_accumulation}"
echo "LR/warmup/min LR:     ${LEARNING_RATE}/${LR_WARMUP_STEPS}/${MIN_LR}"
echo "Max/save steps:       ${MAX_STEPS}/${SAVE_INTERVAL}"
echo "Validation interval:  ${VAL_CHECK_INTERVAL}"
echo "Eval chunks/batch:     $(( 20 * EVAL_CHUNKS_PER_EPISODE ))/20 global (5 per GPU)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q SFT_OUTPUT_ROOT=%q bash %q' \
    "${CUDA_VISIBLE_DEVICES}" "${RUN_ROOT}" "${SCRIPT_DIR}/run_sft.sh"
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${RUN_ROOT}"
GPU_METRICS_PATH="${RUN_ROOT}/gpu_metrics.csv"
export CUDA_VISIBLE_DEVICES
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits \
  --loop=1 > "${GPU_METRICS_PATH}" &
monitor_pid=$!

stop_gpu_monitor() {
  if kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap stop_gpu_monitor EXIT INT TERM

SFT_OUTPUT_ROOT="${RUN_ROOT}" \
  bash "${SCRIPT_DIR}/run_sft.sh" "${TRAIN_ARGS[@]}"

echo "Training complete: ${RUN_ROOT}"
echo "GPU metrics:       ${GPU_METRICS_PATH}"
