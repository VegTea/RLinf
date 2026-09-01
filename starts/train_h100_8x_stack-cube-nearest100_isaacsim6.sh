#!/usr/bin/env bash

# Train the standard IsaacLab stack-cube task on one node with 8 x H100.
# This keeps the 4-GPU experiment's global batch and total environment count
# unchanged while distributing them across eight workers.
# Examples:
#   MAX_EPOCHS=5 bash starts/train_h100_8x_stack-cube_isaacsim6.sh
#   bash starts/train_h100_8x_stack-cube_isaacsim6.sh algorithm.update_epoch=2

set -euo pipefail

STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${STARTS_DIR}/source.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPUS[@]}" -ne 8 ]]; then
    echo "ERROR: expected exactly 8 GPUs in CUDA_VISIBLE_DEVICES, got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi
for gpu in "${VISIBLE_GPUS[@]}"; do
    if [[ -z "${gpu}" ]]; then
        echo "ERROR: CUDA_VISIBLE_DEVICES contains an empty GPU entry: ${CUDA_VISIBLE_DEVICES}" >&2
        exit 2
    fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
    HOST_GPU_COUNT="$(nvidia-smi -L | wc -l)"
    if (( HOST_GPU_COUNT < 8 )); then
        echo "ERROR: requested 8 GPUs, but nvidia-smi exposes only ${HOST_GPU_COUNT}" >&2
        exit 2
    fi
fi

MAX_EPOCHS="${MAX_EPOCHS:-1000}"
if [[ ! "${MAX_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_EPOCHS must be a positive integer, got: ${MAX_EPOCHS}" >&2
    exit 2
fi

export LOG_NAME_TAG="${LOG_NAME_TAG:-h100-8x-stack-cube-nearest100-isaacsim6-chunkopt}"

# Reusing a four-GPU Ray head would silently leave half the GPUs unavailable.
# Refuse that state instead of stopping a potentially user-owned Ray cluster.
if ray status >/dev/null 2>&1; then
    RAY_GPU_COUNT="$("${RLINF_MODEL_PYTHON}" - <<'PY' | tail -n 1
import ray

ray.init(address="auto", logging_level="ERROR")
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)"
    if [[ "${RAY_GPU_COUNT}" != "8" ]]; then
        echo "ERROR: the active Ray cluster exposes ${RAY_GPU_COUNT} GPUs, expected 8." >&2
        echo "Stop/recreate that Ray cluster explicitly before launching this job." >&2
        exit 2
    fi
else
    ray start --head --num-gpus=8
fi

exec bash examples/embodiment/run_embodiment.sh \
    isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
    "${ROBOT_PLATFORM:-LIBERO}" \
#    runner.logger.experiment_name=isaaclab_ppo_openpi_pi05_stack_cube_nearest100_h100_8x \
#    runner.max_epochs="${MAX_EPOCHS}" \
#    env.train.total_num_envs=128 \
#    env.eval.total_num_envs=32 \
#    actor.micro_batch_size=32 \
#    actor.global_batch_size=256 \
    "$@"
