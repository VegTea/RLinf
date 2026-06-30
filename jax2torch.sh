source .venv/bin/activate

python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint-dir /inspire/qb-ilm/project/gjjproject/czxs24230043/checkpoints/pi05_insert-mouse-battery_mixed/pi05_insert-mouse-battery_mixed_2h200/199999 \
  --config-name pi05_yam \
  --output-path checkpoints/torch/yam_pi05_insert_mouse_battery_199999 \
  --precision bfloat16