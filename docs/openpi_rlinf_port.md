# OpenPI_RLinf 并存移植与验证

## 来源与范围

- 上游：`https://github.com/RLinf/RLinf.git`。
- 基线：`b023deb7e90ceadfe18e0a528e862e3c630f7e0a`（删除旧 OpenPI / 改名前）。
- 后续补丁：`184ba07f0a5b3a09a4696a4e3c8aa4b3c720fa86` 的融合 kernel 编译开关、SFT loader、norm-stats 路径、horizon 校验及相应配置。
- **不执行上游的删除旧实现与重命名**。`openpi` 保持旧路径；`openpi_rlinf` 为新增路径。
- 新模型、数据加载、LeRobot 路径辅助、权重转换器、Pi0/Pi0.5 模板、RoboTwin/BEHAVIOR/双 Franka SFT 配方及文档已移入。
- 新数据配置和 policy transforms 独立放在 `rlinf/models/embodiment/openpi_rlinf/{dataconfig,policies,data_transforms}`，避免修改旧 PPO 的数据语义。
- 不移植上游整体模型注册器/调度器重构，不切换现有 PPO，不开发 DINOv2。模型 factory 仍需要预训练权重；“从零训练新架构”是后续任务。
- 上游 RL/DAgger/DSRL/RLT 相关模型模块作为依赖源文件保留，但本次只接入并验证新模型的普通 SFT 路径；这些附加算法的配置与 worker 迁移不在本次范围。

## 本地适配

共享代码仅增加新模型分支、FSDP 配置校验、`openpi_cosine` 学习率调度和新路径的数据集 epoch 计数；旧 OpenPI 的调用和配置保持不变。SFT shell 入口增加 Hydra 参数透传与真实退出码。

上游新的 `hybrid_engines/fsdp` YAML 引用改为本仓库现有的 `training_backend/fsdp`。新 dataconfig/policy 路径隔离；RoboCasa helper 的导入适配为本地目录。导入新的 SFT worker 分支，不整份覆盖上游共享 worker。

### FSDP 检查点修复

在 PyTorch 2.7.1 / FSDP1 / NO_SHARD 上，Pi0.5 最后一个 prefix 分支输出不参与 action loss，会导致部分嵌套 FSDP backward hook 不执行。真实 GPU 测试最初在导出 state_dict 时因缺失最后一层 prefix RMSNorm `scale` 失败。

`openpi_rlinf/pi0.py::compute_loss` 给损失附加 `0 * prefix_out.float().sum()`：损失值不变，但该分支梯度变为零而非 None，使 FSDP 完成 backward 清理并可保存/恢复。注意 AdamW 对零梯度参数仍执行 weight decay，区别于跳过 None 梯度参数；这是有记录的新模型兼容适配，不修改旧 PPO 模型。GPU 回归测试保留了这个失败场景。

## 测试结果（2026-09-23）

### 新模型 SFT

```bash
.venv-model312/bin/python -m pytest tests/unit_tests/test_openpi_rlinf_sft.py -q
# CPU: 3 passed, 1 skipped（CUDA 测试需显式开启）

CUDA_VISIBLE_DEVICES=1 OPENPI_RLINF_CUDA_TEST=1 \
  .venv-model312/bin/python -m pytest tests/unit_tests/test_openpi_rlinf_sft.py -q -s
# 在启智 xhc-test-4090-copy 内：4 passed
```

GPU 测试使用缩小维度的**真实** SigLIP、双专家 Gemma 和 flow-matching loss，不使用假模型：

- 选择性 bf16/fp32 参数与嵌套 FSDP1 NO_SHARD。
- 前向、反向、裁剪、AdamW 更新；确认权重改变。
- DCP model-state 导出与加载，以及 optimizer state 保存/恢复。
- CPU 还验证模型 factory 从 safetensors/full_weights 重新构建模型、损失复现、非法动作维度、horizon、配置与 epoch 上限。
- 一次 GPU 通过记录：loss `2.378250`，grad norm `17.442987`，checkpoint restored。
- 日志：`logs/openpi_rlinf_port_validation/sft_gpu_fixed.log`，退出码文件为 `0`。

检查点转换 CLI 的 `--help`、相关 Python 文件的 Ruff 检查、shell 语法与 `git diff --check` 同时通过。

这不是全尺寸 Pi0.5 数据集收敛实验，不重新证明上游 JAX 对齐结果，也不代表附加算法已验证。

### 旧 IsaacLab PPO 回归

实例：启智 `xhc-test-4090-copy`，2 × RTX 4090（每卡报告 49140 MiB）。
驱动 `595.71.05`（支持 CUDA 13.2）。**实际 Python runtime**：模型环境
`torch 2.7.1+cu126`，Isaac Sim 环境 `torch 2.10.0+cu128`；没有升级这两个共享环境。
当前实例未被释放，因此无需创建替代实例。若重建，应按用户要求选择 CUDA 13.2 compute-stack 镜像，不能仅凭 nvidia-smi 的驱动支持版本判断镜像 runtime。

在实例内、仓库根目录执行：

```bash
source starts/source.sh
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export LOG_NAME_TAG=openpi-rlinf-port-ppo-smoke
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05 LIBERO \
  runner.max_epochs=2 runner.val_check_interval=-1 runner.save_interval=-1 \
  env.train.total_num_envs=2 env.eval.total_num_envs=1 \
  env.train.max_steps_per_rollout_epoch=20 \
  algorithm.rollout_epoch=1 algorithm.update_epoch=1 \
  actor.micro_batch_size=2 actor.global_batch_size=8 \
  runner.logger.experiment_name=openpi_rlinf_port_ppo_smoke
```

Ray 实际发现两张 4090；配置的 `placement: all` 启动两组 actor/rollout/env rank。
启动脚本曾设置 CUDA_VISIBLE_DEVICES=0，但 RLinf 根据硬件发现重新分配 worker，实际使用两卡；因此复现时不应将它误认为单卡试验。

结果：

| 迭代 | policy loss | actor grad norm | value loss |
| --- | ---: | ---: | ---: |
| 1/2 | -0.365 | 1115.1 | 0.053 |
| 2/2 | -0.107 | 756.9 | 0.040 |

- 两轮真实仿真 rollout → GAE → actor/critic 参数更新 → 下一轮权重同步完成，退出码 **0**。
- 原始日志：`logs/openpi_rlinf_port_validation/ppo.log`。
- launcher 产物：`logs/20260923-04:46:54-openpi-rlinf-port-ppo-smoke/`。
- 这是运行兼容性 smoke test，短 rollout 的零成功率不用于判断策略效果。
- 不保存全尺寸 PPO checkpoint，未验证长期收敛或全尺寸 PPO 断点续训。
- 验证后训练进程已退出，GPU 占用恢复测试前水平；未终止实例内原有任务。

### nearest100 单步追加验证（2026-09-23）

随后在同一双卡 4090 实例实际运行了
`isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100`，不是基础叠方块配置。
保留原配置的 `scenario_reset.enabled=true`、`mode=random` 和 nearest100 JSONL
路径（文件包含 100 条场景），仅通过命令行缩小测试工作量：

```bash
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100 LIBERO \
  runner.max_epochs=1 runner.val_check_interval=-1 runner.save_interval=-1 \
  env.train.total_num_envs=2 env.eval.total_num_envs=2 \
  env.train.max_steps_per_rollout_epoch=50 \
  algorithm.rollout_epoch=1 algorithm.update_epoch=1 \
  actor.micro_batch_size=2 actor.global_batch_size=20 \
  runner.logger.experiment_name=nearest100_1step_validation
```

结果：Global Step **1/1**，退出码 **0**；采样、GAE、actor/critic 更新完成。
训练循环一步约 65 秒（不含启动/加载）；policy loss `-0.044`、grad norm `203.0`、
value loss `0.053`，均为有限值。短 rollout 成功率为 0，不作为策略质量判断。
没有修改原训练 YAML，不代表完整 450 步、多轮收敛或保存恢复已验证。

- 完整可复现脚本：`logs/nearest100_1step_validation/run.sh`
- 日志与退出码：`logs/nearest100_1step_validation/run.log`、`exit_code`
- 原始运行产物：`logs/20260923-05:58:09-nearest100-1step-validation/`

## 保留与限制

- 本次没有提交或推送 git commit。
- 导入前已有的 37 个改动/文件删除，按 SHA256 或缺失状态核对全部保持原样。
- `rlinf/models/embodiment/openpi/` 和已跟踪的 `examples/embodiment/config/` 对 HEAD 无 diff。
- 新 SFT 的 full-scale 数据加载仍需用户提供数据、tokenizer、norm stats 和模型权重。
- 官方无限 OpenPI loader 不提供精确逐样本恢复；stateful loader 才保存额外数据位置。
- 上游 `Pi0.sft_forward` 会关闭 gradient checkpointing；不要假定打开 YAML 开关即可降低显存。

使用说明：`docs/source-zh/rst_source/examples/embodied/sft_openpi_rlinf.rst`。
