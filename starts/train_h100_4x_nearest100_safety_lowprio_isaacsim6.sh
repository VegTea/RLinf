#!/usr/bin/env bash
# Re-submit this command with the SAME absolute run root after preemption.
# Model: RLinf-pi05-SFT-Stack-cube; safety env; seed 42; offline W&B.
set -euo pipefail
if [[ $# -lt 1 || "$1" != /* ]]; then
    echo "Usage: bash $0 /absolute/persistent/run-root [Hydra overrides...]" >&2
    exit 2
fi
RUN_ROOT="$1"
shift
STARTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STARTS_DIR/source.sh"
mkdir -p "$RUN_ROOT/attempts"
# Prevent two GPU jobs from writing/pruning the same checkpoint tree.
exec 9> "$RUN_ROOT/train.lock"
flock -n 9 || { echo "Another attempt owns $RUN_ROOT" >&2; exit 75; }
ATTEMPT="$(mktemp -d "$RUN_ROOT/attempts/$(date +%Y%m%d-%H%M%S).XXXXXX")"
printf '%s\n' "$ATTEMPT" > "$RUN_ROOT/.active_attempt.$$"
mv "$RUN_ROOT/.active_attempt.$$" "$RUN_ROOT/active_attempt"
exec > >(tee -a "$ATTEMPT/run_embodiment.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$ATTEMPT/exit_status"' EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE=offline WANDB_ENTITY=vegtea-hust WANDB_PROJECT=rlinf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export ROBOT_PLATFORM=LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
unset WANDB_API_KEY WANDB_RUN_ID
printf 'Attempt started %s on %s; checkpoint root %s\n' "$(date -Iseconds)" "$(hostname)" "$RUN_ROOT/checkpoints"
nvidia-smi --query-gpu=index,name,driver_version --format=csv
python -c 'import torch; assert torch.cuda.device_count() == 4, torch.cuda.device_count()'
if ray status >/dev/null 2>&1; then
    echo 'Refusing to reuse an existing Ray cluster; launch in a fresh job container.' >&2
    exit 2
fi
ray start --head --num-gpus=4 --disable-usage-stats
python examples/embodiment/train_embodied_agent.py \
    --config-path "$EMBODIED_PATH/config" \
    --config-name isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100_instanceable_safety_lowprio_4gpu \
    runner.logger.log_path="$ATTEMPT" \
    runner.checkpoint_dir="$RUN_ROOT/checkpoints" \
    "$@"
