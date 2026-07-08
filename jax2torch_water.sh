source .venv/bin/activate

python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint-dir /inspire/qb-ilm/project/gjjproject/czxs24230043/checkpoints/pi05_seal-water-bottle-cap_mixed/pi05_seal-water-bottle-cap_mixed_2h200/120000 \
  --config-name pi05_yam \
  --output-path checkpoints/torch/yam_pi05_seal_water_bottle_cap_120000 \
  --precision bfloat16
