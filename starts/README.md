# 实验启动脚本

本目录保存可复现的本地实验命令和启动脚本。模型及实验配置放在 Hydra YAML 中；启动脚本只负责选择配置、准备运行环境变量和传递 Hydra 覆盖参数。

不要在这些文件中写入访问令牌、密码或临时 TensorBoard 代理链接。

## IsaacLab checkpoint 评测与视频保存

评测 `.pt` checkpoint，为每个环境保存一段外部／桌面视角与腕部视角横向拼接的视频，同时输出：

- `eval_results.json`：评测结果。
- `eval_scenarios.jsonl`：可重新加载的重置后场景设置。
- `eval_throughput.json`：成功环境统计首次成功步数，失败环境统计完整回合长度。

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh \
  /path/to/full_weights.pt \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  /path/to/eval-output \
  env.eval.total_num_envs=32
```

省略 checkpoint 参数会使用默认基础配置的 `rollout.model.model_path`：

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh
```

用 `--scenario-file PATH` 加载已有 JSONL。脚本保留所选 YAML 的 `mode` 和
`loop`；未传该参数时完全沿用 YAML 的重置行为：

```bash
bash starts/eval_isaaclab_checkpoint_videos.sh \
  --scenario-file /path/to/scenarios.jsonl \
  --config-name isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  env.eval.total_num_envs=32
```

## 8 张 H200 的 nearest100 VLM LoRA 训练

在仓库根目录启动按卡数线性扩展的 8 卡训练；Ray 必须已识别全部 8 张 H200：

```bash
bash starts/train_h200_8x_nearest100_vlm_lora.sh
```

默认使用 256 个训练环境、actor 微批大小 64、actor 全局批大小 512、
VLM LoRA 秩 32。末尾追加的 Hydra 参数优先于默认值：

```bash
bash starts/train_h200_8x_nearest100_vlm_lora.sh \
  runner.max_epochs=10 \
  actor.model.lora_rank=16
```

## 定期保存与离线 W&B

所有 `starts/train*.sh` 支持以下环境变量：

- `SAVE_INTERVAL=100`：每 100 个训练步(默认)保存一次常规 checkpoint，现有 YAML 默认值 40。
  一个训练步是一次采样加 PPO 更新，不是仿真步或优化器步。正数间隔也会在训练结束时保存；
  `-1` 关闭定期保存，但不关闭最佳模型保存。
- `USE_WANDB=false`：默认只用 TensorBoard；设为 `true` 或 `1` 时额外启用 W&B。
  脚本强制设置 `WANDB_MODE=offline`，即使外部设置了在线模式也会覆盖，无需登录 W&B。
  若模型 Python 环境尚未安装可选依赖 `wandb`，需先安装。

```bash
SAVE_INTERVAL=20 USE_WANDB=true \
  bash starts/train_h100_8x_stack-cube-nearest100-instanceable_isaacsim6.sh
```

命令末尾的 Hydra 参数优先于脚本默认值，例如 `runner.save_interval`、
`runner.logger.logger_backends` 和 `++runner.save_best_checkpoint=false`。
通过 `runner.logger.logger_backends=[tensorboard,wandb]` 启用 W&B 也保持离线。
W&B 数据保存在新运行的 `wandb/` 目录；开启独立进程日志时也会写入各进程日志目录。
需要上传时手动执行 `wandb sync /path/to/offline-run-*`。

## 按训练成功率保存最佳模型

这些启动脚本默认开启同步 `EmbodiedRunner` 的 `runner.save_best_checkpoint`；
其他入口和配置需自行开启。该功能要求 `runner.weight_sync_interval=1`
（这些脚本的现有默认值），否则报错，避免把未经测量的 actor 权重误存为最佳模型。

监控指标是每轮日志中的聚合**训练成功率 `env/success_once`**，不是 `eval/success_once`。
每次指标严格提高且为有限值时，保存完整 actor checkpoint（权重、优化器及学习率调度器状态）：

```text
<log_path>/<experiment_name>/checkpoints/best/global_step_<N>/actor/
<log_path>/<experiment_name>/checkpoints/best/best.json
```

- 指标缺失、非有限值或与最佳值相同时，不替换最佳模型。
- `best.json` 记录指标名、成功率、训练步数和 checkpoint 路径。
- 最佳模型在**采样指标收集完成后、对应 PPO 更新前**保存，确保权重对应测得的成功率。
  `N` 是该策略已完成的更新次数，也是指标的日志迭代编号；首次可能保存 `global_step_0`。
- 常规 `checkpoints/global_step_<N>` 仍在更新后保存，与最佳模型独立。
- 新最佳模型和索引成功写入后，才删除当前运行创建的上一份最佳模型。
  替换期间需容纳两份最佳模型；成功率提高会增加保存开销。

### 从最佳模型续训

`runner.resume_dir` 指向 `actor` 的上一级目录，建议用绝对路径。
启动脚本会自动创建新的运行日志目录：

```bash
bash starts/train_h100_8x_stack-cube-nearest100-instanceable_isaacsim6.sh \
  runner.resume_dir=/absolute/path/checkpoints/best/global_step_520
```

常规和最佳 checkpoint 都包含记录最佳分数的 `runner_state.json`。
恢复时还会检查原运行的 `best/best.json`，因此从旧的常规 checkpoint 续训也会保留原运行较新的最佳分数。
新运行只有超过该分数才保存新最佳模型，且不会删除原运行的 checkpoint，请保留原目录。
没有这些元数据的旧 checkpoint 会从首次有效指标开始记录最佳值。

此功能不恢复环境、随机数生成器或课程学习状态；续训会重新采样，并非精确恢复仿真现场。

### IK 诊断（单环境或批量）

向评测脚本追加
`++env.eval.init_params.ik_diagnostic_path=/absolute/path/ik_trace.jsonl`。
支持多个环境，每条记录附带本 worker 的 `env_id` 和实际 `scenario_id`；
多个 worker 自动使用 `.rank<rank>` 文件后缀，避免相互覆盖。
批量固定场景评测使用 sequential、loop=true、freeze_per_env=true，
并核验导出的场景 ID 集合与输入一致，避免初始化/自动 reset 消耗场景造成遗漏。
默认关闭；诊断文件必须不存在，避免覆盖旧证据。每个物理子步记录原始环境动作、缩放后动作、
目标/实际 TCP、关节角/速度/限位、原有 IK 输出目标、Jacobian 及奇异值、三个方块位置；
每个控制步结束另存实际状态。TCP/Jacobian 为机器人基座坐标系，方块为环境局部坐标系，
四元数保留当前 IsaacLab 的原生顺序。步号从 1 开始，时间由 metadata 的 physics_dt 和 decimation 换算。
观察现有控制器调用，不额外求解 IK 或修改动作；同步写盘会降低吞吐，不适合性能测试。
Jacobian 混合米/弧度单位，奇异值只适合在同一控制定义下比较，不能单独证明碰撞或 OOD。
单条场景重复 reset 的评测需 `env.eval.init_params.scenario_reset.loop=true`。

CPU 检查：`PYTHONPATH=. .venv-model312/bin/python tests/unit_tests/test_isaaclab_ik_diagnostics.py`。

批量评测结束后使用 `.venv-isaacsim6/bin/python toolkits/isaaclab/summarize_joint_limits.py
--run <输出目录> --scenarios <原始场景JSONL>` 生成 `joint_limit_report.csv` 和
`joint_limit_summary.json`。只统计七个手臂关节，不把夹爪正常开闭端点计为故障。
接近限位阈值 0.01 rad，触限阈值 0.001 rad，IK 越界阈值 0.001 rad；
“持续顶限”要求同一关节实际触限且 IK 目标继续朝同侧越界，连续至少 0.5 秒。
另外统计顶限时的位置跟踪误差 >1 cm、指令下降但实际上升的控制步数。
限位事件与任务失败只表示共现，不直接证明因果。分析前强制检查场景唯一覆盖、
450 个控制步的完整记录以及每个结果对应视频存在。`--self-test` 可单独检查分类边界。
