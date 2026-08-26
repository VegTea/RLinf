# RLinf IsaacLab Smoke Train 与环境分析总结

本文整理此前工作的第 4～7 部分，包括 IsaacLab smoke train、视频保存问题、当前任务的相机设置，以及 `nearest100` 场景的域随机化情况。

## 4. IsaacLab Smoke Train

我们先后在 RTX 4090 节点和 H100 任务中尝试 smoke train，目标是验证：

- Isaac Sim 能正常启动；
- RTX/Vulkan 渲染链路正常；
- Ray 的 worker 和 GPU placement 正常；
- actor、rollout、env 之间能够联通；
- PPO 至少完成一次短 rollout 和 update；
- 保存 10 条 rollout 视频。

H100 任务使用的镜像为：

```text
docker.sii.shaipower.online/inspire-studio/rlinf-xhc:1.3
```

任务使用最高优先级。由于平台提供的单卡 H100 quota 与计算组不匹配，曾采用申请同机双卡、但仅让训练进程使用 GPU 0 的方式：

```bash
export CUDA_VISIBLE_DEVICES=0
ray start --head --num-gpus=1
```

同时，将 RLinf 的 actor、env 和 rollout 限制在 hardware rank 0：

```text
cluster.component_placement={actor\,env\,rollout:0}
```

### 已遇到的问题

此前的 smoke train 失败主要包括：

1. eval 环境数量与 RLinf/Ray 检测到的 GPU 或 env worker 数量不匹配；
2. Hydra 命令行中的 placement mapping 转义错误；
3. `component_placement` 被 Hydra 解析成字符串，而不是映射；
4. 视频编码阶段找不到可用的 FFmpeg executable。

这些错误分别发生在配置校验、worker placement 或视频编码阶段，不能全部归因于 Isaac Sim、CUDA 或 Vulkan。

### 当前状态

目前尚未确认最终 10 条训练过程视频已经成功生成。因此，“完成一次 smoke train 并保存 10 条视频”仍是待完成项。

预期视频目录形式为：

```text
logs/gpu_smoke_video/<run-directory>/video/eval/seed_42/
```

预期包含：

```text
0.mp4
1.mp4
...
9.mp4
```

## 6. 当前 IsaacLab 环境与相机设置

任务使用 Franka Panda 机械臂完成三块积木堆叠：

```text
将红色方块放到蓝色方块上，再将绿色方块放到红色方块上。
```

环境 ID 为：

```text
Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0
```

基础环境配置位于：

```text
examples/embodiment/config/env/isaaclab_stack_cube.yaml
```

### 相机

任务不是只有一个外部相机，而是包含两个相机：

- `table_cam`：桌面外部相机，提供第三人称视角；
- `wrist_cam`：安装在 Franka 末端附近的腕部相机，提供第一人称视角。

两个相机的分辨率均为 `256 × 256`。环境能够产生：

```text
table_cam_rgb
table_cam_depth
wrist_cam_rgb
wrist_cam_depth
```

当前 OpenPI/pi0.5 配置主要使用两路 RGB 图像，即桌面相机 RGB 和腕部相机 RGB。环境虽然生成深度观测，但不代表深度图一定会进入当前策略模型。

在 `nearest100` 的场景 reset 中：

- `table_cam` 的位姿可以由场景记录指定，但当前 100 条记录中的变化非常小；
- `wrist_cam` 相对于机械臂末端的安装位姿固定；
- `wrist_cam` 会随着 Franka 末端运动而自然改变世界坐标中的观察视角。

## 7. `nearest100` 场景与域随机化

训练配置为：

```text
examples/embodiment/config/
isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100.yaml
```

训练和 eval 都启用了：

```yaml
scenario_reset:
  enabled: true
  mode: random
```

每次 reset 会从 `nearest100` JSONL 文件中随机选择场景。实际应用的内容包括：

- 替换桌子 USD 资产；
- 设置三个方块的位置和姿态；
- 设置桌面相机的位置和旋转。

### 场景统计

`nearest100` 文件共有 100 条记录，其主要多样性如下：

| 项目 | 数量或占比 |
| --- | ---: |
| 场景记录 | 100 |
| 桌子 USD 资产 | 6 种 |
| `table_base.usd` | 81 条 |
| `table_Graphite.usd` | 14 条 |
| 其余 4 种桌子 | 各 1～2 条 |
| 每个方块的位置组合 | 约 34 种 |
| `table_cam` 位置 | 1 种 |
| `table_cam` 旋转 | 2 种，差异很小 |

`table_base.usd` 和 `table_Graphite.usd` 是同一类 Seattle Lab 桌子的不同外观资产：

- `table_base.usd`：默认桌子外观；
- `table_Graphite.usd`：石墨灰或深灰外观。

它们主要改变颜色和材质，不代表桌子的尺寸、形状、碰撞几何或工作空间发生明显变化。81 条 `table_base.usd` 记录也不是 81 种外观，而是同一个桌子资产在不同场景记录中重复出现。

### 实际没有启用的大范围视觉随机化

IsaacLab 任务代码中预定义了以下随机化接口：

- 多种 HDR 环境光纹理；
- 灯光强度随机化；
- 灯光颜色随机化；
- 多种桌面纹理；
- 多种机器人金属纹理。

但是当前配置下大部分没有真正启用，原因包括：

- 环境默认使用 `eval_mode = False`；
- `eval_type = None`；
- 启用 `scenario_reset` 后，原有的桌面纹理随机化事件被替换成 `noop_event`；
- JSONL 中的 `source_cube_id`、`source_camera_id` 等主要是来源元数据，当前 reset 逻辑不会根据 `source_cube_id` 更换方块外观。

因此，当前实际生效的主要变化是：

```text
少量桌子颜色/材质资产变化
+ 方块初始位置和朝向变化
+ 极小的外部相机位姿变化
```

它不能算大范围视觉域随机化。并且训练和 eval 使用同一个 `nearest100` 文件，因此当前 eval 也不属于严格的视觉 OOD 泛化测试。

## 8. 双卡 RTX 4090 与双卡 H100 训练配置

以下配置都以同一个基础 YAML 为准：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100.yaml
```

不建议复制整份 YAML。使用命令行 override 更容易保证模型路径、场景路径和算法参数与基础配置保持一致。

### 8.1 共同启动准备

在仓库根目录执行：

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
```

在启动训练前，需要确认 Ray 看到的是当前两张卡。如果当前没有 Ray 集群：

```bash
ray start --head --num-gpus=2
ray status
```

如果节点上已经存在 Ray，不能直接假设它捕获了正确的 `CUDA_VISIBLE_DEVICES`；应先检查 Ray 状态和资源数量。训练日志开头应显示：

```text
1 node and 2 accelerators
hardware ranks: [[0], [1]]
```

基础 YAML 已包含：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all
```

因此双卡正式训练不需要额外覆盖 `component_placement`。

### 8.2 双卡 RTX 4090 配置

这里的建议针对此前日志中检测到的约 48 GiB RTX 4090 节点。单卡实测在 `2 env + micro_batch_size=4` 时峰值约为 `39.7 GiB / 49.1 GiB`。

建议从每卡 4 个训练环境开始，即总共 8 个环境：

```yaml
env:
  train:
    total_num_envs: 8
  eval:
    total_num_envs: 2
    video_cfg:
      save_video: false

actor:
  micro_batch_size: 4
  global_batch_size: 240
  enable_offload: false

rollout:
  enable_offload: true
```

完整启动命令：

```bash
LOG_NAME_TAG=nearest100-2x4090 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=8 \
  env.eval.total_num_envs=2 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=4 \
  actor.global_batch_size=240
```

该设置保留基础配置中的正式 PPO 参数：

```yaml
algorithm:
  rollout_epoch: 2
  update_epoch: 3
  gamma: 0.99
  gae_lambda: 0.95
```

如果稳定运行且每卡仍有至少 5～6 GiB 余量，可以依次测试：

```text
8 env → 12 env → 16 env
```

一次只调整 `env.train.total_num_envs`，不要同时放大 env 数和 micro batch，否则无法判断显存变化来自哪里。

注意：普通消费级 RTX 4090 通常是 24 GiB。此前节点日志报告的是约 49 GiB 可见显存，因此上述配置不能直接视为“任意 24 GiB 4090 都能运行”。对于真正的 24 GiB 4090，当前 OpenPI actor、critic、Isaac 渲染和 rollout 全部共卡的配置很可能 OOM，需要重新实测 offload 或调整 placement。

### 8.3 双卡 H100 80GB 配置

双 H100 实测结果为：

| 总训练 env | micro batch | 峰值显存/卡 | 结果 |
| ---: | ---: | ---: | --- |
| 64 | 8 | 约 40.9 GiB | PASS |
| 128 | 8 | 约 51.3 GiB | PASS |
| 192 | 8 | 约 61.5 GiB | PASS |
| 256 | 8 | 约 72.2 GiB | PASS |
| 256 | 32 | 约 74 GiB | PASS |

`256 env + micro 32` 虽然通过，但距离 80GB 上限较近，不适合作为未经长期运行验证的默认正式配置。建议正式训练从总共 192 个 env 开始：

```yaml
env:
  train:
    total_num_envs: 192
  eval:
    total_num_envs: 2
    video_cfg:
      save_video: false

actor:
  micro_batch_size: 32
  global_batch_size: 256
  enable_offload: false

rollout:
  enable_offload: true
```

完整启动命令：

```bash
LOG_NAME_TAG=nearest100-2xh100 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=192 \
  env.eval.total_num_envs=2 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256
```

如果目标是追求吞吐量，可以在 192 env 稳定后测试 224 和 256 env。对于 256 env，建议持续监控显存，因为短 smoke test 中峰值已经达到约 74 GiB/卡，长期训练还需要为验证、日志、checkpoint 和显存碎片保留空间。

### 8.4 显存与训练语义注意事项

当前基础配置为：

```yaml
actor:
  fsdp_config:
    sharding_strategy: no_shard
```

这意味着两张卡各自保留完整模型副本。双卡主要增加并行吞吐量和可承载的总环境数，并不会把模型显存简单减半。

4090 配置使用 `global_batch_size=240`，H100 配置使用 `global_batch_size=256`。这是因为 actor 更新要求每卡 rollout 数量可以被每卡 global batch 整除，同时 global batch 还必须能被 `micro_batch_size × actor_world_size` 整除。actor buffer 按策略生成的 5-step action chunk 记录决策，而不是把每个物理仿真 step 都作为一条独立策略样本。对 8 个环境、450 step、5-step action chunk、2 个 rollout epoch 而言，总训练样本数为 `8 × 450 / 5 × 2 = 1440`，因此 240 是满足约束且接近原始 256 的取值。`micro_batch_size` 只控制一次前向/反向处理的数据量：

- 4090 使用 `micro_batch_size=4`，显存较稳，但梯度累积次数更多；
- H100 使用 `micro_batch_size=32`，更新吞吐量更高。

正式训练默认关闭 eval 视频。视频编码和 eval rollout 会引入额外开销，并且需要可用的 FFmpeg。建议通过单独的周期性 eval 任务保存视频，而不是让正式训练持续录制。
