#!/usr/bin/env bash
set -euo pipefail

cd /home/user/Workspace/RLinf

mkdir -p logs/recap_server

PYTHONUNBUFFERED=1 \
TF_CPP_MIN_LOG_LEVEL=2 \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python toolkits/standalone_eval_scripts/openpi/recap_policy_server.py \
  --checkpoint-dir /home/user/Workspace/RLinf/checkpoints/yam_tower_of_hanoi_game_step41000/yam_tower_of_hanoi_game_step41000 \
  --config-name pi05_yam \
  --repo-id assets/tower-of-hanoi-game/expert-success-hil-suffix-mix-data \
  --host 0.0.0.0 \
  --port 8084 \
  --device cuda:0 \
  --action-chunk 50 \
  --action-env-dim 14 \
  --guidance-scale 1.0 \
  2>&1 | tee logs/recap_server/recap-8084.log
