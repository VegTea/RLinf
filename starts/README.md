# Experiment launchers

This directory stores reproducible commands and small launcher scripts for
local experiments. Keep model and experiment settings in Hydra YAML files;
launchers here should only select a config, prepare runtime environment
variables, and pass optional Hydra overrides.

Do not put access tokens, passwords, or temporary TensorBoard proxy URLs in
these files.

## IsaacLab checkpoint evaluation with videos

Evaluate a `.pt` checkpoint and save one horizontally concatenated
external/table-plus-wrist video per environment, plus `eval_results.json`:

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh \
  /path/to/full_weights.pt \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  /path/to/eval-output \
  env.eval.total_num_envs=32
```

省略 checkpoint 参数会使用 nearest100 配置的默认 `rollout.model.model_path`：

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh
```

## Shared environment and 4 x H100 nearest100

The common environment file exports both Python interpreters, the local Isaac
Sim 6 asset root, and the migrated scenario asset root. Source it before
starting Ray, or use the 4-GPU launcher which sources it automatically:

```bash
bash starts/train_h100_4x_nearest100_isaacsim6.sh
```

The launcher defaults to `CUDA_VISIBLE_DEVICES=0,1,2,3`, 128 training
environments, 32 evaluation environments, `micro_batch_size=64`, and
`global_batch_size=256`. Set `MAX_EPOCHS` for a short run and append Hydra
overrides as needed:

```bash
MAX_EPOCHS=3 bash starts/train_h100_4x_nearest100_isaacsim6.sh
```

## Nearest-100 on 1 x RTX 4090 (Isaac Sim 6)

The single-GPU launcher uses the migrated nearest100 JSONL scenario under
`rlinf/assets_isaaclab`, the Python 3.12 model environment, and the separate
Isaac Sim 6 environment. Its conservative defaults are two training
environments, actor micro batch 2, and actor global batch 180.

```bash
bash starts/train_1x4090_nearest100_isaacsim6.sh
```

Set `MAX_EPOCHS` to control the training duration (default: 6000). For a
short validation run:

```bash
MAX_EPOCHS=10 bash starts/train_1x4090_nearest100_isaacsim6.sh
```

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
