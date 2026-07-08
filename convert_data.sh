#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw
OUTPUT_ROOT=/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/recap/phase1
INTERMEDIATE_ROOT=data/recap/phase1_tmp
EVAL_RATIO=0.1
SEED=42

TASKS=(
  insert-mouse-battery
  seal-water-bottle-cap
  tower-of-hanoi-game
)

output_name() {
  case "$1" in
    insert-mouse-battery) echo "insert_mouse_battery_split" ;;
    seal-water-bottle-cap) echo "seal_water_bottle_cap_split" ;;
    tower-of-hanoi-game) echo "tower_of_hanoi_game_split" ;;
    *) echo "${1//-/_}_split" ;;
  esac
}

for TASK_NAME in "${TASKS[@]}"; do
  TASK_OUTPUT="$(output_name "${TASK_NAME}")"
  INTERMEDIATE="${INTERMEDIATE_ROOT}/${TASK_NAME}"
  SPLIT_OUTPUT="${OUTPUT_ROOT}/${TASK_OUTPUT}"

  echo "============================================================"
  echo "Preparing phase1 task: ${TASK_NAME}"
  echo "  raw: ${RAW_ROOT}/${TASK_NAME}"
  echo "  intermediate: ${INTERMEDIATE}"
  echo "  output: ${SPLIT_OUTPUT}"
  echo "============================================================"

  .venv/bin/python examples/recap/process/prepare_yam_recap_view.py \
    --raw-root "${RAW_ROOT}/${TASK_NAME}" \
    --task-name "${TASK_NAME}" \
    --output-root "${INTERMEDIATE}" \
    --expert-dir-name expert-data \
    --success-dir-name success-and-hil-data \
    --failure-dir-name failure-data \
    --expert-output-name expert \
    --success-output-name success_hil \
    --failure-output-name failure \
    --force

  .venv/bin/python examples/recap/process/check_recap_readiness.py \
    --dataset "${INTERMEDIATE}/expert:sft:yam" \
    --dataset "${INTERMEDIATE}/success_hil:sft:yam" \
    --dataset "${INTERMEDIATE}/failure:reward:yam" \
    --failure-reward -2000

  for BUCKET in expert success_hil failure; do
    .venv/bin/python examples/recap/process/split_lerobot_dataset.py \
      --source-root "${INTERMEDIATE}/${BUCKET}" \
      --output-root "${SPLIT_OUTPUT}/${BUCKET}" \
      --eval-ratio "${EVAL_RATIO}" \
      --seed "${SEED}" \
      --mode random \
      --force
  done
done
