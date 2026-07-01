source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 bash examples/recap/value/run_value_sft.sh yam_seal-water-bottle-cap_sft_value \
  runner.max_steps=2 \
  runner.val_check_interval=-1 \
  runner.save_interval=100 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=1 \
  data.train_num_workers=0 \
  data.eval_num_workers=0