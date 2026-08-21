# IsaacLab H100 Scaling Report

## Run

- qzcli job: `job-1b4e70ad-56b9-4c81-a807-aeb68679782f`
- Image: `docker.sii.shaipower.online/inspire-studio/rlinf-xhc:1.3`
- Hardware: one node, 2 x NVIDIA H100 80GB, driver 570.124.06, CUDA 12.8
- Config: `isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100`
- Rollout: 450 steps, global batch 576, one rollout/update step per point
- All points in the final run produced `exit_status=0` and `SMOKE_TEST_PASS`.
- qzcli was invoked with `--priority 10` (the CLI maximum). The platform's
  task detail normalizes this to internal `priority=20`, `priority_name=4`,
  `priority_level=NORMAL`; this same normalization appears on earlier jobs
  submitted with qzcli priority 10, so the effective queue band is controlled
  by workspace policy rather than the CLI label alone.

The first submitted job was stopped after `qb-prod-gpu045` reproduced
`VkResult: ERROR_DEVICE_LOST`. The final job excluded that node. The failure
was node-level Isaac/Vulkan instability, not an OOM or PPO failure.

## Measured Results

Peak memory is the maximum of the two GPUs; each H100 reports 81,559 MiB.

| sweep | total envs | envs/GPU | micro batch | step seconds | env steps/s | peak MiB/GPU |
|---|---:|---:|---:|---:|---:|---:|
| env | 64 | 32 | 8 | 280.5 | 102.674 | 41,114 |
| env | 128 | 64 | 8 | 493.2 | 116.788 | 51,304 |
| env | 192 | 96 | 8 | 730.1 | 118.340 | 61,951 |
| env | 256 | 128 | 8 | 964.7 | 119.415 | 72,614 |
| micro | 256 | 128 | 4 | 1,223.8 | 94.133 | 72,943 |
| micro | 256 | 128 | 8 | 967.7 | 119.045 | 72,167 |
| micro | 256 | 128 | 16 | 870.2 | 132.383 | 73,063 |
| micro | 256 | 128 | 32 | 814.6 | 141.419 | 72,919 |
| representative | 256 | 128 | 32 | 801.9 | 143.659 | 74,640 |

The representative row ran for two training steps. Its peak is about 91.5%
of H100 memory, leaving about 6.9 GiB per GPU. Micro-batch size mainly changed
training throughput; it did not materially change peak memory in this range.

Raw results are in
`logs/h100_scaling/20260817-20:12:23-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100-2gpu/benchmark_results.csv`.

## 8-GPU Configuration

The directly scalable candidate is:

```yaml
cluster:
  num_nodes: 1
env:
  train:
    total_num_envs: 1024       # 128 per GPU
    max_episode_steps: 450
    max_steps_per_rollout_epoch: 450
actor:
  micro_batch_size: 32
  global_batch_size: 2304      # 2304 % (8 * 32) == 0
```

This preserves the measured 128 env/GPU and 288 samples/GPU global-batch
ratio from the 2-GPU experiment. The measured 2-GPU representative throughput
would ideally scale to roughly 574 env steps/s on 8 GPUs, but this is an
estimate; inter-GPU communication, CPU capacity, and Isaac scheduling must be
measured on the 8-GPU node.

Use `total_num_envs: 768` (96 env/GPU) as the first 8-GPU smoke fallback if
the node has less CPU headroom or if 1024 envs causes initialization pressure.
The 2-GPU data puts that point near the 62 GiB/GPU memory regime, while 1024
envs is expected near the measured 74.6 GiB/GPU regime. Keep `micro_batch_size:
32` initially; reduce it to 16 only if the actor update itself becomes a
bottleneck. Keep the 8-GPU global batch divisible by `8 * micro_batch_size`.

These values are for the benchmark's one-rollout/one-update measurements.
For production, retain the YAML's intended `algorithm.rollout_epoch: 2` and
`algorithm.update_epoch: 3` unless a separate optimization study changes them.
