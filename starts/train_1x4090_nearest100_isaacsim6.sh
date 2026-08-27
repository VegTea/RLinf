#!/usr/bin/env bash

# Train the Isaac Sim 6 nearest100 scenario on one RTX 4090 (48 GiB).
# Additional Hydra overrides are appended, for example:
#   MAX_EPOCHS=10 bash starts/train_1x4090_nearest100_isaacsim6.sh

set -euo pipefail

STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${STARTS_DIR}")"
MODEL_VENV="${REPO_PATH}/.venv-model312"

if [[ ! -x "${MODEL_VENV}/bin/python" ]]; then
    echo "ERROR: Model environment not found: ${MODEL_VENV}/bin/python" >&2
    exit 2
fi

if [[ ! -x "${REPO_PATH}/.venv-isaacsim6/bin/python" ]]; then
    echo "ERROR: Isaac Sim 6 environment not found: ${REPO_PATH}/.venv-isaacsim6/bin/python" >&2
    exit 2
fi

cd "${REPO_PATH}"
source "${MODEL_VENV}/bin/activate"

# The simulator, rollout, and actor share this GPU. Do not silently start a
# multi-GPU run when CUDA_VISIBLE_DEVICES has been inherited from a shell.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPUS[@]}" -ne 1 || -z "${VISIBLE_GPUS[0]}" ]]; then
    echo "ERROR: expected exactly one GPU in CUDA_VISIBLE_DEVICES, got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

export LOG_NAME_TAG="${LOG_NAME_TAG:-nearest100-isaacsim6-1x4090}"
MAX_EPOCHS="${MAX_EPOCHS:-6000}"
if [[ ! "${MAX_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_EPOCHS must be a positive integer, got: ${MAX_EPOCHS}" >&2
    exit 2
fi

# A rollout has 2 envs * (450 / 5) action chunks * 2 rollout epochs = 360
# samples. global_batch_size=180 gives two optimizer batches and is divisible
# by micro_batch_size=2 for a one-rank actor.
exec bash examples/embodiment/run_embodiment.sh \
    isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
    "${ROBOT_PLATFORM:-LIBERO}" \
    runner.logger.experiment_name=isaaclab_ppo_openpi_pi05_table_nearest100_1x4090 \
    runner.max_epochs="${MAX_EPOCHS}" \
    env.train.total_num_envs=2 \
    env.eval.total_num_envs=1 \
    actor.micro_batch_size=2 \
    actor.global_batch_size=180 \
    "$@"
