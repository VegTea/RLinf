# Experiment launchers

This directory stores reproducible commands and small launcher scripts for
local experiments. Keep model and experiment settings in Hydra YAML files;
launchers here should only select a config, prepare runtime environment
variables, and pass optional Hydra overrides.

Do not put access tokens, passwords, or temporary TensorBoard proxy URLs in
these files.

## Nearest-100 VLM LoRA on 8 x H200

Start the linearly scaled 8-GPU training run from the repository root (Ray
must already expose all eight H200 GPUs):

```bash
bash starts/train_h200_8x_nearest100_vlm_lora.sh
```

The defaults use 256 training environments, actor micro batch 64, actor global
batch 512, and VLM LoRA rank 32. Additional Hydra overrides are appended, so
they take precedence over these defaults:

```bash
bash starts/train_h200_8x_nearest100_vlm_lora.sh \
  runner.max_epochs=10 \
  actor.model.lora_rank=16
```

## Nearest-100 table-copper VLM LoRA

Start training from the repository root:

```bash
bash starts/train_nearest100_table_copper_vlm_lora.sh
```

Pass additional Hydra overrides after the script name:

```bash
bash starts/train_nearest100_table_copper_vlm_lora.sh \
  runner.max_epochs=10 \
  runner.save_interval=2
```

Start TensorBoard for the latest matching run:

```bash
bash starts/tensorboard_nearest100_table_copper_vlm_lora.sh
```

An explicit log directory can be supplied as the first argument:

```bash
bash starts/tensorboard_nearest100_table_copper_vlm_lora.sh \
  logs/20260825-14:49:17-nearest100-vlm-lora-4x4090-formal
```
