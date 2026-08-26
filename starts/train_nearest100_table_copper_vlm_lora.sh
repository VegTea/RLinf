#!/usr/bin/env bash

# 启动 nearest100 table-copper 场景上的 OpenPI VLM LoRA 正式训练。
#
# 默认使用 4 张 GPU（0,1,2,3）。可以在命令前覆盖环境变量，例如：
#   CUDA_VISIBLE_DEVICES=0,1 bash starts/train_nearest100_table_copper_vlm_lora.sh
#
# 传给本脚本的所有参数都会继续传给 Hydra，可用于临时覆盖配置，例如：
#   bash starts/train_nearest100_table_copper_vlm_lora.sh runner.max_epochs=10

set -euo pipefail

# 根据脚本位置计算仓库根目录，因此可以从任意工作目录启动。
STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${STARTS_DIR}")"

# 必须使用仓库的 Python 3.11 虚拟环境。直接使用系统或 Conda Python
# 可能与 Isaac Sim 自带的 Python 标准库不兼容，并触发 SRE module mismatch。
if [[ ! -x "${REPO_PATH}/.venv/bin/python" ]]; then
    echo "ERROR: Python environment not found: ${REPO_PATH}/.venv/bin/python" >&2
    exit 2
fi

cd "${REPO_PATH}"

# 将 .venv 放在 PATH 首位，确保 run_embodiment.sh source Isaac Sim 环境后
# 仍然调用仓库虚拟环境中的 python。
export PATH="${REPO_PATH}/.venv/bin:${PATH}"

# 默认使用本机 4 张 RTX 4090；调用方可以通过同名环境变量覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# run_embodiment.sh 会用该标签生成带时间戳的日志目录。
export LOG_NAME_TAG="${LOG_NAME_TAG:-nearest100-vlm-lora-4x4090-formal}"

# LIBERO 在这里是 OpenPI 的机器人平台/动作归一化标识；实际环境仍由
# Hydra 配置 nearest100_table_copper_vlm_lora.yaml 指定为 IsaacLab。
exec bash examples/embodiment/run_embodiment.sh \
    nearest100_table_copper_vlm_lora \
    "${ROBOT_PLATFORM:-LIBERO}" \
    "$@"
