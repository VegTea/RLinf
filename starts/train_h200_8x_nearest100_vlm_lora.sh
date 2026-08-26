#!/usr/bin/env bash

# 在单机 8 x H200 上复现 nearest100 实验，并通过 LoRA 微调 OpenPI 的 VLM。
#
# 默认值从以下 4 x H100 实验线性扩展而来：
#   logs/20260823-18:07:39-h100-4x-nearest100-env128-micro64-500step-20260823-p4
#
# 额外 Hydra 参数可以覆盖下面的默认值，例如：
#   bash starts/train_h200_8x_nearest100_vlm_lora.sh runner.max_epochs=10

set -euo pipefail

STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${STARTS_DIR}")"

if [[ ! -x "${REPO_PATH}/.venv/bin/python" ]]; then
    echo "ERROR: Python environment not found: ${REPO_PATH}/.venv/bin/python" >&2
    exit 2
fi

cd "${REPO_PATH}"
export PATH="${REPO_PATH}/.venv/bin:${PATH}"

# Ray 必须已经在这 8 张卡上启动。允许调用方选择其他 8 张卡，但不允许
# 静默退化成少于 8 个 actor rank，否则 global_batch_size 的约束会改变。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPUS[@]}" -ne 8 ]]; then
    echo "ERROR: expected exactly 8 entries in CUDA_VISIBLE_DEVICES, got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

export LOG_NAME_TAG="${LOG_NAME_TAG:-h200-8x-nearest100-env256-micro64-vlm-lora-500epoch}"

# 4 -> 8 卡线性扩展：env 128 -> 256，global batch 256 -> 512；保持每卡
# micro batch=64、每卡环境数=32，以及每次 rollout 的 optimizer step 数不变。
# OpenPI 的 VLM LoRA 需要同时跳过 freeze_vlm()，并让 FSDP 保留原始参数，
# 才能正确处理冻结的 VLM 基座参数和可训练的 LoRA 参数混合。
exec bash examples/embodiment/run_embodiment.sh \
    isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
    "${ROBOT_PLATFORM:-LIBERO}" \
    runner.logger.experiment_name=isaaclab_ppo_openpi_pi05_table_nearest100_h200_8x_vlm_lora \
    runner.max_epochs=500 \
    env.train.total_num_envs=256 \
    env.eval.total_num_envs=64 \
    actor.micro_batch_size=64 \
    actor.global_batch_size=512 \
    actor.model.is_lora=true \
    actor.model.lora_rank=32 \
    actor.model.openpi.train_expert_only=false \
    actor.fsdp_config.use_orig_params=true \
    "$@"
