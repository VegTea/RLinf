source .venv/bin/activate

export CFG_LOG_ROOT=/inspire/qb-ilm/project/gjjproject/public/xhc/phase2
export YAM_GENERALIST_PI05_TORCH_CKPT=/inspire/qb-ilm/project/gjjproject/public/xl/rss-challenge/checkpoints/pi05_rss_generalist_450000/torch

bash examples/recap/cfg/run_cfg_sft.sh yam_rss_generalist_cfg_openpi \
  data.num_workers=12 \
  actor.micro_batch_size=16 \
  actor.global_batch_size=128 \
  runner.val_check_interval=1000 \
  actor.optim.total_training_steps=150000 \
  runner.max_steps=150000 \
  runner.save_interval=30000
