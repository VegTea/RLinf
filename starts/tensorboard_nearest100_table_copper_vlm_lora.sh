#!/usr/bin/env bash

# 为 nearest100 table-copper VLM LoRA 实验启动 TensorBoard。
#
# 不传参数时，脚本自动选择最近一次匹配的正式训练日志目录：
#   bash starts/tensorboard_nearest100_table_copper_vlm_lora.sh
#
# 也可以显式指定日志目录（支持相对仓库根目录的路径或绝对路径）：
#   bash starts/tensorboard_nearest100_table_copper_vlm_lora.sh logs/<run-dir>
#
# 默认监听 0.0.0.0:6006，可通过环境变量覆盖：
#   TENSORBOARD_PORT=6007 bash starts/tensorboard_nearest100_table_copper_vlm_lora.sh

set -euo pipefail

# 根据脚本位置计算仓库根目录，因此可以从任意工作目录启动。
STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${STARTS_DIR}")"
LOG_DIR="${1:-}"

# TensorBoard 与训练使用同一个仓库虚拟环境。
if [[ ! -x "${REPO_PATH}/.venv/bin/tensorboard" ]]; then
    echo "ERROR: TensorBoard not found: ${REPO_PATH}/.venv/bin/tensorboard" >&2
    exit 2
fi

# 未显式指定日志时，按照目录修改时间选择最新的正式训练目录。
if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="$({
        find "${REPO_PATH}/logs" \
            -maxdepth 1 \
            -type d \
            -name '*-nearest100-vlm-lora-4x4090-formal' \
            -printf '%T@ %p\n'
    } | sort -nr | head -1 | cut -d' ' -f2-)"
# 将调用方传入的相对路径解释为相对仓库根目录，而不是当前工作目录。
elif [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${REPO_PATH}/${LOG_DIR}"
fi

# 避免 TensorBoard 因空路径而意外扫描当前目录或其他无关目录。
if [[ -z "${LOG_DIR}" || ! -d "${LOG_DIR}" ]]; then
    echo "ERROR: no matching experiment log directory found." >&2
    exit 2
fi

echo "Starting TensorBoard for ${LOG_DIR}"

# 使用 0.0.0.0 才能通过启智开发机的端口代理访问。
exec "${REPO_PATH}/.venv/bin/tensorboard" \
    --logdir "${LOG_DIR}" \
    --host "${TENSORBOARD_HOST:-0.0.0.0}" \
    --port "${TENSORBOARD_PORT:-6006}"
