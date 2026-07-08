source .venv/bin/activate

export CFG_LOG_ROOT=/inspire/qb-ilm/project/gjjproject/public/xhc/phase2
export YAM_HANOI_PI05_TORCH_CKPT=/inspire/ssd/project/gjjproject/czxs24230043/RLinf/checkpoints/torch/yam_pi05_tower_of_hanoi_game_199999

bash examples/recap/cfg/run_cfg_sft.sh yam_tower-of-hanoi-game_cfg_openpi \
  data.num_workers=12 \
  actor.micro_batch_size=16 \
  actor.global_batch_size=128 \
  runner.val_check_interval=1000 \
  actor.optim.total_training_steps=50000 \
  runner.max_steps=50000 \
  runner.save_interval=10000
