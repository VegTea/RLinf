source .venv/bin/activate

bash examples/recap/value/run_value_sft.sh yam_insert_mouse_battery_sft_value \
  +runner.resume_dir=/inspire/ssd/project/gjjproject/czxs24230043/RLinf/logs/value_sft/yam_insert_mouse_battery_sft_value-20260630-14:40:03/yam_insert_mouse_battery_value_sft/checkpoints/global_step_8000 \
  runner.max_steps=16000 \
  runner.val_check_interval=500 \
  runner.save_interval=1000 \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256 \
  actor.optim.total_training_steps=16000 \
  data.train_num_workers=6 \
  data.eval_num_workers=2
