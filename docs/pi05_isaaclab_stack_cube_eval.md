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

不传参数时，脚本默认使用 nearest100 配置的
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
- `eval_results.json`：汇总所有 env 的单个 JSON，逐项记录上述 setting 和结果；
- `eval_metrics.pt` 和 `eval_embodiment.log`：聚合指标和完整日志。

脚本强制 `algorithm.eval_rollout_epoch=1`，确保一个视频只对应当前 env 的一次
完整评测。其他 Hydra 覆盖项可以从第四个参数开始继续追加。

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

- `env.eval.total_num_envs=32` 决定输出 32 个视频。每个环境的常规评测长度为
  450 个控制 step，与 GPU 数量无关。
- 保留 `wait_for_video_writes=true`。默认视频编码是异步的；如果不等待写盘完成，
  Ray 退出时可能会截断最后一个 MP4。
- `camera_keys` 与 `per_env_videos` 依赖本次为 `RecordVideo` wrapper 增加的支持。
  删除其中任一个会恢复为常规的单个合并视频行为。
- `eval_embodiment.sh` 默认将 `ISAAC_PATH` 设为 `${REPO_PATH}/isaac_sim`。只有
  Isaac Sim 安装在其他位置时，才需要在启动前显式设置 `ISAAC_PATH`。
- 传给 PI0.5 的指令由任务配置决定，不在 shell 命令中设置：
  `Stack the red block on the blue block, then stack the green block on the red block.`
