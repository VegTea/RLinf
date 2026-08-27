#!/usr/bin/env bash

# Train nearest100 on one node with 4 x H100 and Isaac Sim 6.
# Extra arguments are passed to Hydra, for example:
#   MAX_EPOCHS=5 bash starts/train_h100_4x_nearest100_isaacsim6.sh

set -euo pipefail

STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${STARTS_DIR}/source.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Using NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}"
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPUS[@]}" -ne 4 || -z "${VISIBLE_GPUS[0]}" || -z "${VISIBLE_GPUS[1]}" || -z "${VISIBLE_GPUS[2]}" || -z "${VISIBLE_GPUS[3]}" ]]; then
    echo "ERROR: expected exactly 4 GPUs in CUDA_VISIBLE_DEVICES, got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

MAX_EPOCHS="${MAX_EPOCHS:-6000}"
if [[ ! "${MAX_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_EPOCHS must be a positive integer, got: ${MAX_EPOCHS}" >&2
    exit 2
fi

export LOG_NAME_TAG="${LOG_NAME_TAG:-h100-4x-nearest100}"

# Ray must see the final environment variables and all four visible GPUs.
if ! ray status >/dev/null 2>&1; then
    ray start --head --num-gpus=4
fi

exec bash examples/embodiment/run_embodiment.sh \
    isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
    "${ROBOT_PLATFORM:-LIBERO}" \
    runner.logger.experiment_name=isaaclab_ppo_openpi_pi05_table_nearest100_h100_4x \
    runner.max_epochs="${MAX_EPOCHS}" \
    env.train.total_num_envs=128 \
    env.eval.total_num_envs=32 \
    actor.micro_batch_size=64 \
    actor.global_batch_size=256 \
    "$@"
