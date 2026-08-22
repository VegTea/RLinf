# PI0.5 IsaacLab Stack-Cube 评测

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
