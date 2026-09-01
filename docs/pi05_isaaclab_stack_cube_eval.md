# PI0.5 IsaacLab Stack-Cube 评测

## 指定 checkpoint、逐环境双相机视频和结果 JSON

使用专用启动脚本时，可以在第一个参数传入 `.pt` 权重文件：

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh \
  /path/to/full_weights.pt \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  /path/to/eval-output \
  env.eval.total_num_envs=32
```

不传参数时，脚本默认使用基础 Stack-Cube 配置的
`rollout.model.model_path`（当前为 `RLinf-pi05-SFT-Stack-cube`）作为初始策略：

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh
```

这个默认值是模型目录，不是 `.pt` state dict，因此不会被错误地传入
`runner.ckpt_path`。

输出目录包含：

- `video/eval/seed_*/*_env_*.mp4`：每个 env 一个视频，左侧为外部/桌面相机，
  右侧为腕部相机；
- `per_env_results/*.json`：每个 env/episode 一份结果，包含 `scenario_id`、
  `policy_success`、`checkpoint_path`（若显式提供）或 `policy_source`（默认模型目录）
  和 `video_path`；
- `eval_results.json`：成功/失败数量、成功率，以及成功和失败视频的相对路径；
- `eval_throughput.json`：按有效环境 step 计算的成功吞吐量。成功环境使用首次
  成功 step，失败环境使用完整 episode 长度；
- `eval_scenarios.jsonl`：每个实际评测场景一行，可再次传给
  `--scenario-file`；每行使用唯一的 `eval_XXXXXX` ID，输入场景原 ID 保存在
  `source_scenario_id`；
- `eval_metrics.pt` 和 `eval_embodiment.log`：聚合指标和完整日志。

脚本强制 `algorithm.eval_rollout_epoch=1`，确保一个视频只对应当前 env 的一次
完整评测。其他 Hydra 覆盖项可以从第四个参数开始继续追加。

例如让 32 个环境各运行至多 600 个控制 step，需要同时覆盖 episode 截断长度和
评测 rollout 长度：

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh \
  env.eval.total_num_envs=32 \
  env.eval.max_episode_steps=600 \
  env.eval.max_steps_per_rollout_epoch=600
```

吞吐量定义为
`num_success / sum(first_success_step if success else episode_len)`。例如三个环境
分别在第 200、350 step 成功，第三个到第 450 step 仍失败，则结果为
`2 / (200 + 350 + 450) = 0.002 success/step`，也就是每 1000 env-steps 成功 2 次。
并行运行只改变墙钟耗时，不改变这个分母；成功环境仍按现有逻辑运行到统一 horizon，
不会因该指标提前 reset。

## 指定或保存场景

显式加载场景文件：

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh \
  --scenario-file /path/to/eval_scenarios.jsonl \
  --config-name isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  env.eval.total_num_envs=32
```

`--scenario-file` 只替换配置中的 `scenario_file` 并启用 scenario reset；采样
方式仍由所选 YAML 的 `mode` 和 `loop` 决定。`sequential` 从文件开头顺序取，
`random` 从整个文件随机取。输入 JSONL 会在启动仿真前校验必需字段、向量长度和
ID 唯一性。

不传 `--scenario-file` 时，不会强行开启 scenario reset：nearest100 配置继续
加载其 YAML 中的场景文件，基础配置则使用 IsaacLab 原生随机 reset。两种情况下
都会从 reset 后的实际仿真状态导出恰好 `env.eval.total_num_envs` 行，包括桌子资产、
三个 cube 的位置/RPY 和 table camera 位姿。

## 使用预训练模型目录评测

这是已验证可运行的 nearest-100 Stack-Cube PI0.5 独立评测命令。该配置中已经
显式指定模型目录，无需额外传入 checkpoint 路径。

```bash
cd "$WORK_ROOT/RLinf-IsaacLab-Diverse-PPO/RLinf"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard

bash examples/embodiment/eval_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  algorithm.eval_rollout_epoch=1 \
  env.eval.total_num_envs=32 \
  env.eval.video_cfg.save_video=true \
  env.eval.video_cfg.per_env_videos=true \
  env.eval.video_cfg.wait_for_video_writes=true \
  "env.eval.video_cfg.camera_keys=[main_images,wrist_images]"
```

启动时会打印结果目录，例如 `logs/<timestamp>/`。其 `video/eval/` 目录中有
32 个 MP4，每个环境对应一个视频。视频每帧水平拼接：左侧为外部/桌面相机，右侧为
腕部相机。聚合指标及逐轨迹指标保存在 `logs/<timestamp>/eval_metrics.pt`。

## 注意事项

- `env.eval.total_num_envs=32` 决定输出 32 个视频。默认每个环境评测 450 个控制
  step；修改长度时必须同时覆盖 `max_episode_steps` 和
  `max_steps_per_rollout_epoch`，与 GPU 数量无关。
- 保留 `wait_for_video_writes=true`。默认视频编码是异步的；如果不等待写盘完成，
  Ray 退出时可能会截断最后一个 MP4。
- `camera_keys` 与 `per_env_videos` 依赖本次为 `RecordVideo` wrapper 增加的支持。
  删除其中任一个会恢复为常规的单个合并视频行为。
- `eval_embodiment.sh` 默认将 `ISAAC_PATH` 设为 `${REPO_PATH}/isaac_sim`。只有
  Isaac Sim 安装在其他位置时，才需要在启动前显式设置 `ISAAC_PATH`。
- 传给 PI0.5 的指令由任务配置决定，不在 shell 命令中设置：
  `Stack the red block on the blue block, then stack the green block on the red block.`
