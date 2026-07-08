source .venv/bin/activate

export CFG_LOG_ROOT=/inspire/qb-ilm/project/gjjproject/public/xhc/phase2
export YAM_MOUSE_PI05_TORCH_CKPT=/inspire/ssd/project/gjjproject/czxs24230043/RLinf/checkpoints/torch/yam_pi05_insert_mouse_battery_80000

ADVANTAGE_TAG=insert_mouse_battery
MOUSE_DATASETS=(
  /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/recap/phase1/insert_mouse_battery_split/expert/train
  /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/recap/phase1/insert_mouse_battery_split/success_hil/train
  /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/recap/phase1/insert_mouse_battery_split/failure/train
  /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/recap/phase2/insert_mouse_battery_hil_split/train
)

missing=0
for dataset in "${MOUSE_DATASETS[@]}"; do
  advantage_file="${dataset}/meta/advantages_${ADVANTAGE_TAG}.parquet"
  if [ ! -f "${advantage_file}" ]; then
    echo "Missing advantage file: ${advantage_file}" >&2
    missing=1
  fi
done

if [ "${missing}" -ne 0 ]; then
  echo "Run compute_advantages_yam_mouse successfully before CFG training." >&2
  exit 1
fi

bash examples/recap/cfg/run_cfg_sft.sh yam_insert_mouse_battery_cfg_openpi \
  data.num_workers=12 \
  actor.micro_batch_size=16 \
  actor.global_batch_size=128 \
  runner.val_check_interval=1000 \
  actor.optim.total_training_steps=50000 \
  runner.max_steps=50000 \
  runner.save_interval=10000
