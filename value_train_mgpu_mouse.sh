source .venv/bin/activate

TASK_NAME=insert_mouse_battery bash examples/recap/value/run_value_sft.sh yam_insert_mouse_battery_sft_value \
  runner.max_steps=4000 \
  runner.val_check_interval=50 \
  runner.save_interval=1000 \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256 \
  actor.model.freeze_vlm=false \
  actor.model.freeze_vision_encoder=false \
  actor.fsdp_config.use_orig_params=true \
  actor.optim.lr=2.5e-5 \
  actor.optim.value_lr=5.0e-5 \
  actor.optim.lr_warmup_steps=100 \
  actor.optim.total_training_steps=4000 \
  data.train_num_workers=6 \
  data.eval_num_workers=2
