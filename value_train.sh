source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 bash examples/recap/value/run_value_sft.sh yam_insert_mouse_battery_sft_value \
  runner.max_steps=8000 \
  runner.val_check_interval=-1 \
  runner.save_interval=1000 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=16 \
  actor.optim.total_training_steps=8000 \
  data.train_num_workers=4 \
  data.eval_num_workers=0