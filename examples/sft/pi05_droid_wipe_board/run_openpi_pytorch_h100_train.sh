#!/usr/bin/env bash

# Four-H100 production recipe for the JAX-aligned RLinf OpenPI implementation.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != 4 )); then
  echo "Exactly four visible GPUs are required: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

for gpu_id in "${GPU_IDS[@]}"; do
  gpu_name="$(nvidia-smi --id="${gpu_id//[[:space:]]/}" --query-gpu=name --format=csv,noheader | head -n 1)"
  if [[ "${gpu_name}" != *H100* && "${ALLOW_NON_H100:-0}" != "1" ]]; then
    echo "Expected H100, found ${gpu_id}: ${gpu_name}" >&2
    exit 1
  fi
done

RUN_TIMESTAMP="$(date -u +'%Y%m%d-%H%M%S')"
RUN_ROOT="${H100_TRAIN_ROOT:-${REPO_ROOT}/outputs/pi05_droid_openpi_pytorch_h100/${RUN_TIMESTAMP}}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
EVAL_CHUNKS_PER_EPISODE="${EVAL_CHUNKS_PER_EPISODE:-10}"

ARGS=(
  "runner.max_steps=${MAX_STEPS}"
  "runner.save_interval=${SAVE_INTERVAL}"
  "runner.val_check_interval=${VAL_CHECK_INTERVAL}"
  "data.episode_split.eval_chunks_per_episode=${EVAL_CHUNKS_PER_EPISODE}"
  "actor.micro_batch_size=${MICRO_BATCH_SIZE:-16}"
  "actor.global_batch_size=${GLOBAL_BATCH_SIZE:-64}"
  "actor.eval_batch_size=${EVAL_BATCH_SIZE:-5}"
)
ARGS+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q SFT_OUTPUT_ROOT=%q bash %q' \
    "${CUDA_VISIBLE_DEVICES}" "${RUN_ROOT}" "${SCRIPT_DIR}/run_openpi_pytorch_sft.sh"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${RUN_ROOT}"
export CUDA_VISIBLE_DEVICES
SFT_OUTPUT_ROOT="${RUN_ROOT}" bash "${SCRIPT_DIR}/run_openpi_pytorch_sft.sh" "${ARGS[@]}"
