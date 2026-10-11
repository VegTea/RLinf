#!/usr/bin/env bash

# Shared by train*.sh before starting Ray. CLI Hydra overrides take precedence.
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
USE_WANDB="${USE_WANDB:-false}"
if [[ ! "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ && "${SAVE_INTERVAL}" != "-1" ]]; then
    echo "ERROR: SAVE_INTERVAL must be a positive integer or -1 (disabled)." >&2
    return 2
fi
case "${USE_WANDB,,}" in
    true|1) TRAIN_LOGGER_BACKENDS='[tensorboard,wandb]' ;;
    false|0) TRAIN_LOGGER_BACKENDS='[tensorboard]' ;;
    *) echo "ERROR: USE_WANDB must be true/false or 1/0." >&2; return 2 ;;
esac
# Also covers W&B selected by a trailing logger_backends Hydra override.
export WANDB_MODE=offline
TRAIN_OPTIONS=(
    "runner.save_interval=${SAVE_INTERVAL}"
    "runner.logger.logger_backends=${TRAIN_LOGGER_BACKENDS}"
    "++runner.save_best_checkpoint=true"
)
