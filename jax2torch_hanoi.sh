source .venv/bin/activate

python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint-dir /inspire/qb-ilm/project/gjjproject/czxs24230043/checkpoints/pi05_tower-of-hanoi-game_mixed/pi05_tower-of-hanoi-game_mixed_2h200/199999 \
  --config-name pi05_yam \
  --output-path checkpoints/torch/yam_pi05_tower_of_hanoi_game_199999 \
  --precision bfloat16
