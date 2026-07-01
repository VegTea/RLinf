TASK_NAME=tower-of-hanoi-game

.venv/bin/python examples/recap/process/prepare_yam_recap_view.py \
  --raw-root /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw/${TASK_NAME} \
  --task-name ${TASK_NAME} \
  --output-root data/recap/${TASK_NAME} \
  --expert-dir-name expert-data \
  --success-dir-name success-and-hil-data \
  --failure-dir-name failure-data \
  --expert-output-name expert \
  --success-output-name success_hil \
  --failure-output-name failure \
  --force

.venv/bin/python examples/recap/process/check_recap_readiness.py \
  --dataset data/recap/${TASK_NAME}/expert:sft:yam \
  --dataset data/recap/${TASK_NAME}/success_hil:sft:yam \
  --dataset data/recap/${TASK_NAME}/failure:reward:yam \
  --failure-reward -2000

.venv/bin/python examples/recap/process/split_lerobot_dataset.py \
  --source-root data/recap/${TASK_NAME}/expert \
  --output-root data/recap/${TASK_NAME}_split/expert \
  --eval-ratio 0.05 \
  --mode tail \
  --force

.venv/bin/python examples/recap/process/split_lerobot_dataset.py \
  --source-root data/recap/${TASK_NAME}/success_hil \
  --output-root data/recap/${TASK_NAME}_split/success_hil \
  --eval-ratio 0.05 \
  --mode tail \
  --force

.venv/bin/python examples/recap/process/split_lerobot_dataset.py \
  --source-root data/recap/${TASK_NAME}/failure \
  --output-root data/recap/${TASK_NAME}_split/failure \
  --eval-ratio 0.05 \
  --mode tail \
  --force