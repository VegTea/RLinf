# RLinf IsaacLab 多 GPU 训练交接文档

本文供后续会话在**单机 4 卡或 8 卡**的 RTX 4090、H100、H200 节点上启动以下任务：

```text
IsaacLab Franka Stack Cube + OpenPI pi0.5 + PPO actor-critic
```

基础配置：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100.yaml
```

仓库：

```text
/inspire/hdd/global_user/czxs24230043/RLinf-IsaacLab-Diverse-PPO/RLinf
```

模型：

```text
/inspire/hdd/global_user/czxs24230043/pretrained_models/RLinf/RLinf-pi05-SFT-Stack-cube
```

推荐镜像：

```text
docker.sii.shaipower.online/inspire-studio/rlinf-xhc:1.3
```

本文只覆盖单机多卡。多机训练还需要在每台机器启动 Ray 前设置不同的 `RLINF_NODE_RANK`，不能直接照搬本文命令。

---

## 1. 已验证事实与尚未验证部分

### 已实测

双 H100 80GB、CUDA 12.8 的 smoke 数据：

| 总 env | 每卡 env | micro batch | 每卡峰值显存 | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 64 | 32 | 8 | 约 40.9 GiB | PASS |
| 128 | 64 | 8 | 约 51.3 GiB | PASS |
| 192 | 96 | 8 | 约 61.5 GiB | PASS |
| 256 | 128 | 8 | 约 72.2 GiB | PASS |
| 256 | 128 | 32 | 约 74 GiB | PASS |

双 H100 的上述测试完成了环境启动、scenario reset、模型加载、action rollout、环境交互和一次 actor update。

单张平台 4090 的日志显示约 49 GiB 可见显存。在以下短 smoke 配置中：

```text
2 env
30 steps
micro batch 4
global batch 12
```

峰值约为 `39.7 GiB / 49.1 GiB`，测试通过。

### 不是实测结论

- 4/8 卡 4090 尚未完成正式长训练验证；
- 4/8 卡 H100 尚未按本文推荐正式配置长时间运行；
- H200 配置是根据显存容量和已有 H100 数据给出的保守起点，必须重新 smoke；
- 不能由双卡测试直接断言 8 卡吞吐线性增长，Isaac 渲染、CPU、Ray 通信和共享存储都可能成为瓶颈。

后续会话必须保持这个区分，不要把推算参数写成“已经验证”。

---

## 2. 基础训练语义

基础 PPO 参数保持不变：

```yaml
algorithm:
  rollout_epoch: 2
  update_epoch: 3
  adv_type: gae
  loss_type: actor_critic
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  value_clip: 0.2
```

环境参数：

```yaml
env:
  train:
    max_episode_steps: 450
    max_steps_per_rollout_epoch: 450
```

模型每次策略决策产生 5 个 action chunk：

```yaml
actor:
  model:
    num_action_chunks: 5
```

actor、rollout 和 Isaac 环境默认共置在所有 GPU 上：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all
```

基础配置当前明确覆盖为：

```yaml
actor:
  fsdp_config:
    sharding_strategy: no_shard
```

`no_shard` 意味着每张 GPU 都保留完整模型副本。增加卡数主要提高数据并行吞吐和可承载的总环境数，**不会把单卡模型显存除以 GPU 数量**。

不要为了让 24GB 卡勉强运行就直接改成 `full_shard`。这会改变模型通信和生命周期行为，需要单独验证 OpenPI、共置 rollout 和权重同步，不能视作普通显存参数。

---

## 3. 最重要的 batch 整除约束

设：

- `W`：actor GPU 数，即单机可见卡数；
- `E`：`env.train.total_num_envs`；
- `S`：`max_steps_per_rollout_epoch`，当前为 450；
- `C`：`num_action_chunks`，当前为 5；
- `R`：`algorithm.rollout_epoch`，当前为 2；
- `M`：`actor.micro_batch_size`；
- `G`：`actor.global_batch_size`。

当前任务每次 PPO update 的 actor buffer 样本数为：

```text
T = E × (S / C) × R
  = E × 90 × 2
  = 180E
```

必须同时满足：

```text
E % W == 0
G % (M × W) == 0
T % G == 0
```

否则常见结果是在环境和 rollout 已经运行很久之后，actor update 才触发断言，例如：

```text
global_batch_size is not divisible by micro_batch_size * world_size
```

或：

```text
rollout_size is not divisible by batch_size_per_rank
```

### 基础 YAML 不能直接原样用于正式 update

基础 YAML 当前是：

```text
E=32, S=450, C=5, R=2, G=256
```

对应：

```text
T = 32 × 90 × 2 = 5760
5760 % 256 != 0
```

因此在 4/8 卡任务里不要只改 `CUDA_VISIBLE_DEVICES` 就直接正式训练，必须覆盖 env/global/micro batch。

### 参数检查脚本

提交任务前可执行：

```bash
python - <<'PY'
W = 8       # GPU 数
E = 256     # total_num_envs
S = 450
C = 5
R = 2
M = 32      # micro batch
G = 256     # global batch

assert S % C == 0, (S, C)
total_samples = E * (S // C) * R
assert E % W == 0, (E, W)
assert G % (M * W) == 0, (G, M, W)
assert total_samples % G == 0, (total_samples, G)
print(
    f"valid: GPUs={W}, total_envs={E}, envs_per_gpu={E // W}, "
    f"samples_per_update={total_samples}, micro={M}, global={G}, "
    f"grad_accum={G // (M * W)}, minibatches={total_samples // G}"
)
PY
```

修改任何一个 env、step、chunk、rollout epoch、micro batch、global batch 或 GPU 数后，都应重新检查。

---

## 4. 根据设备选择初始配置

以下是**首轮正式配置的建议起点**，目标是留出显存和稳定性余量，不是追求极限 env 数。

| 设备 | W | E（总 env） | 每卡 env | M | G | 依据 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4×4090，约 48GB/卡 | 4 | 8 | 2 | 4 | 288 | 沿用单卡已测 env 压力，推算配置 |
| 8×4090，约 48GB/卡 | 8 | 16 | 2 | 4 | 288 | 沿用单卡已测 env 压力，推算配置 |
| 4×H100 80GB | 4 | 128 | 32 | 32 | 256 | 每卡 env 压力低于双卡高负载实测 |
| 8×H100 80GB | 8 | 256 | 32 | 32 | 256 | 每卡 env 压力低于双卡高负载实测 |
| 4×H200，约 141GB/卡 | 4 | 256 | 64 | 32 | 256 | H100 数据外推，必须 smoke |
| 8×H200，约 141GB/卡 | 8 | 512 | 64 | 32 | H100 数据外推，必须 smoke |

表中所有组合均满足当前正式配置 `S=450, C=5, R=2` 的 batch 整除条件。

### 4090 显存必须以实际检测为准

普通消费级 RTX 4090 通常不是此前日志中显示的约 49 GiB 配置。启动任务后必须检查：

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

如果实际只有约 24 GiB：

- 不要直接运行上表中的正式配置；
- 当前 `no_shard`、actor/rollout/env 共卡配置很可能连模型更新阶段都无法容纳；
- 可以从 `E=W`、`M=1`、短 30-step smoke 和 `actor.enable_offload=true` 尝试；
- 即使 smoke 通过，也必须完成一次 actor update 才能说明模型、梯度和优化器能同时容纳；
- 如果仍 OOM，应换 48GB/80GB 卡，或把 FSDP sharding/offload 当作单独工程问题验证，而不是继续把 env 调到 0 附近。

### H100 不建议直接使用极限值

双 H100 上 `128 env/卡 + micro 32` 短测试峰值已约 74 GiB。长训练还会受到显存碎片、validation、checkpoint 和偶发峰值影响，因此建议从 32 env/卡开始。

稳定后可测试：

```text
32 env/卡 → 64 env/卡 → 96 env/卡 → 128 env/卡
```

但 env 数也会增加每次 PPO update 的样本总量和单 step 时长，不是越大越好。

### H200 先做可比测试，再使用额外显存

H200 的额外显存允许更多环境或更大的 micro batch，但 Isaac 仿真可能先受 CPU、渲染或调度限制。建议先用与 H100 相同的 32 env/卡做基线，再尝试本文表中的 64 env/卡。

如果 GPU 利用率没有提升而 step time 继续增加，不应仅因显存还有余量就继续增加 env。

---

## 5. 各设备正式训练命令

### 共同准备

在仓库根目录执行：

```bash
source .venv/bin/activate
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
```

`ROBOT_PLATFORM=LIBERO` 是当前 OpenPI action normalization 所需的平台标识，不表示当前运行的是 LIBERO 环境；实际环境仍是 IsaacLab。

设置可见 GPU 后再启动 Ray。新任务容器中，如果没有既有 Ray：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --num-gpus=4
ray status
```

8 卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head --num-gpus=8
ray status
```

如果 Ray 已经启动，先确认它是否在正确的 `CUDA_VISIBLE_DEVICES` 下启动。只有当该 Ray 会话确实属于当前任务时，才可以停止并重启；不要终止其他用户或其他训练使用的 Ray。

训练日志必须显示正确的 accelerator 数量和 hardware ranks，不能只相信 `CUDA_VISIBLE_DEVICES`。

### 4×约 48GB RTX 4090

```bash
LOG_NAME_TAG=nearest100-4x4090 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=8 \
  env.eval.total_num_envs=4 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=4 \
  actor.global_batch_size=288
```

稳定后可把总 env 从 8 增加到 16；`G=288` 仍满足整除条件。

### 8×约 48GB RTX 4090

```bash
LOG_NAME_TAG=nearest100-8x4090 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=16 \
  env.eval.total_num_envs=8 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=4 \
  actor.global_batch_size=288
```

稳定后可把总 env 从 16 增加到 32；`G=288` 仍满足整除条件。

### 4×H100 80GB

```bash
LOG_NAME_TAG=nearest100-4xh100 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=128 \
  env.eval.total_num_envs=4 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256
```

稳定后优先测试总 env 192、256。保持 `G=256` 时，总 env 应选择 64 的倍数。

### 8×H100 80GB

```bash
LOG_NAME_TAG=nearest100-8xh100 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=256 \
  env.eval.total_num_envs=8 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256
```

稳定后可测试总 env 320、384、512。不要默认 512 一定比 256 吞吐更好。

### 4×H200

```bash
LOG_NAME_TAG=nearest100-4xh200 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=256 \
  env.eval.total_num_envs=4 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256
```

### 8×H200

```bash
LOG_NAME_TAG=nearest100-8xh200 \
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  LIBERO \
  env.train.total_num_envs=512 \
  env.eval.total_num_envs=8 \
  env.eval.video_cfg.save_video=false \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256
```

H200 两套命令必须先通过短 smoke，再恢复正式 450-step 配置。

---

## 6. 每种新设备都应执行的 smoke 流程

不要第一次就在未知节点上运行 450 step × 2 rollout epoch × 3 update epoch。

### 第一步：硬件和驱动检查

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
ls -l /etc/vulkan/icd.d/nvidia_icd.json
```

确认：

- GPU 数和申请数量一致；
- 型号与预期一致；
- 每卡显存一致；
- CUDA 12.8 容器能看到宿主驱动；
- Vulkan NVIDIA ICD 存在。

### 第二步：短 smoke，必须包含 actor update

4×约 48GB 4090 示例：

```bash
bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 4 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  --train-envs 8 \
  --eval-envs 4 \
  --max-episode-steps 30 \
  --rollout-epoch 1 \
  --update-epoch 1 \
  --actor-global-batch 48 \
  --actor-micro-batch 4 \
  --log-root logs/device_smoke/4x4090
```

这里短 smoke 的样本数为：

```text
8 × (30 / 5) × 1 = 48
```

8×约 48GB 4090：

```bash
bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 8 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  --train-envs 16 \
  --eval-envs 8 \
  --max-episode-steps 30 \
  --rollout-epoch 1 \
  --update-epoch 1 \
  --actor-global-batch 96 \
  --actor-micro-batch 4 \
  --log-root logs/device_smoke/8x4090
```

4×H100/H200 的低压基线：

```bash
bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 4 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  --train-envs 64 \
  --eval-envs 4 \
  --max-episode-steps 30 \
  --rollout-epoch 1 \
  --update-epoch 1 \
  --actor-global-batch 384 \
  --actor-micro-batch 32 \
  --log-root logs/device_smoke/4xhbm
```

8×H100/H200 的低压基线：

```bash
bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 8 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 \
  --train-envs 128 \
  --eval-envs 8 \
  --max-episode-steps 30 \
  --rollout-epoch 1 \
  --update-epoch 1 \
  --actor-global-batch 768 \
  --actor-micro-batch 32 \
  --log-root logs/device_smoke/8xhbm
```

### 第三步：检查结果

成功日志必须包含：

```text
SMOKE_TEST_PASS
```

并检查：

```bash
find logs/device_smoke -name gpu_peak.csv -print
find logs/device_smoke -name gpu_smoke.log -print
```

`gpu_peak.csv` 中每张卡都应有采样记录。建议正式配置峰值不超过实际显存的约 85%～90%，至少保留 8～12 GiB 绝对余量给长训练峰值和碎片；小显存卡则应保留至少 3～5 GiB。

如果 smoke 只完成模型加载和 rollout、没有执行 actor update，不能用它判断正式训练是否会 OOM。

---

## 7. 调参顺序

每次只改变一个维度，并记录 step time、GPU 峰值、GPU 利用率和是否通过。

### 7.1 先调 env 数

固定 micro/global batch，增加 `env.train.total_num_envs`：

```text
低压起点 → 约 1.5 倍 → 约 2 倍
```

每次变更后重新检查 batch 整除条件。

env 数主要影响：

- Isaac Sim 渲染和物理仿真显存；
- 图像 observation 显存；
- rollout buffer 大小；
- CPU 与渲染吞吐；
- 每次 PPO update 的样本总量和 step 时长。

停止增加 env 的信号：

- 显存超过 90%；
- GPU 利用率没有改善；
- `env/run_interact_once` 明显变慢；
- CPU 长时间满载；
- Ray worker 出现超时或心跳问题；
- 单步时间增长快于样本吞吐增长。

### 7.2 再调 micro batch

env 固定后再增加：

```text
4090: 4 → 8（只有显存允许时）
H100/H200: 16 → 32 → 64（64 未验证）
```

提高 micro batch 主要减少梯度累积次数、提高 actor update 吞吐，但会增加 update 阶段峰值显存。

每次调整 M 后必须保证：

```text
G % (M × W) == 0
```

