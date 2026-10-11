OpenPI_RLinf 监督训练（并存移植）
================================

本分支保留原有 ``model_type: openpi``，额外提供 ``model_type: openpi_rlinf``。
前者继续服务现有 IsaacLab PPO；后者采用上游与 JAX 参考实现对齐的
PyTorch Pi0 / Pi0.5 模型。本地冒烟测试不等同于重新验证完整的 JAX 数值对齐。

来源与边界
------------

- 基线：RLinf/RLinf ``b023deb7e90ceadfe18e0a528e862e3c630f7e0a``。
- 补入 ``184ba07f`` 的 SFT 数据、归一化、action horizon 校验及可关闭的
  ``torch.compile`` 更新，但不执行其删除旧 OpenPI / 重命名的操作。
- 新实现的数据配置、策略 transforms 放在 ``rlinf/models/embodiment/openpi_rlinf/``，
  与旧 ``openpi/`` 隔离。没有迁移上游通用 scheduler / 其他模型的大规模重构。
- 新模型携带上游 task 类，但本次接入与验证范围是 SFT；现有 IsaacLab PPO
  仍使用旧模型。不要仅修改 PPO YAML 的 ``model_type`` 来切换实现。
- 本次不实现 DINOv2 策略，也不添加随机初始化入口：模型 factory 仍需要基础权重。

环境与配置
------------

使用独立的模型训练环境。Pi0 网络本身是 PyTorch；LeRobot 数据管线仍依赖
RLinf/OpenPI 的 Python 包、JAX 数据 transforms、tokenizer 和数据资产。
参考已有 OpenPI 安装说明；不要原地升级正在用于 PPO 的共享环境。
BEHAVIOR 数据还需要对应的 LeRobot / 视频读取依赖。仿真器不是离线 SFT 的必要条件。

配置入口：

- ``examples/sft/config/robotwin_sft_openpi_rlinf.yaml``：Pi0 RoboTwin。
- ``examples/sft/config/behavior_sft_openpi_pi05_rlinf.yaml``：Pi0.5 BEHAVIOR。
- ``examples/sft/config/realworld_sft_openpi_pytorch_dual_franka_tcp_rot6d.yaml``：
  双 Franka，文件名沿用上游历史名称，内部使用 ``openpi_rlinf``。

模型模板位于 ``examples/sft/config/model/pi0_rlinf.yaml`` 和
``pi0_5_rlinf.yaml``。所有数据、模型路径均需要填写：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_rlinf \
     actor.model.model_path=/absolute/path/to/new_format_model \
     data.train_data_paths=/absolute/path/to/lerobot_dataset \
     actor.model.openpi_data.norm_stats_path=/absolute/path/to/norm_stats.json

``data.train_data_paths`` 是数据集路径或 LeRobot repo id，不是旧入口使用的缓存根目录。
显式传入的 ``norm_stats_path`` 必须存在；不传则使用 TrainConfig 默认统计量并警告。
新的数据配置注册表在 ``rlinf.models.embodiment.openpi_rlinf.dataconfig``。

训练约束
------------

- FSDP1：``actor.fsdp_config.strategy: fsdp``、``sharding_strategy: no_shard``、
  ``use_orig_params: true``。没有在本分支验证 FSDP2 / full_shard。
- ``precision: null`` 保留上游选择性 bf16 / fp32 参数布局；FSDP
  ``mixed_precision`` 的三个 dtype 均设为 null。全 fp32 可设置对应的三个值为 fp32。
- ``actor.optim.lr_scheduler: openpi_cosine`` 使用非零首步 warmup，匹配上游调度。
- ``actor.model.openpi.torch_compile: false`` 可关闭新模型的融合 kernel 编译；
  若字段尚未在 YAML 中出现，命令行使用 ``+actor.model.openpi.torch_compile=false``。
- 网络 horizon 来自 TrainConfig 或 ``openpi.action_horizon``；官方 LeRobot loader
  的训练窗口来自 TrainConfig。执行 chunk 可以更短，不能把两者混为一谈。
- 新路径的 ``max_epochs`` 按数据集长度计数；``max_steps`` 提供训练步数上限。
  旧 SFT / PPO 的计数行为保持不变。
- 当前上游 ``Pi0.sft_forward`` 会关闭 gradient checkpointing；不要仅凭 YAML
  中的开关估算训练显存。大型模型训练前应实际测量。

权重与恢复
------------

``actor.model.model_path`` 支持新格式 ``model.safetensors`` 或 RLinf
``full_weights.pt``。同目录同时存在时优先 full_weights，并发出警告。
完整训练恢复使用 ``runner.resume_dir=/.../checkpoints/global_step_N``；
仅设置 model_path 不会恢复 optimizer / scheduler。

转换工具保留在 ``rlinf/utils/ckpt_convertor/openpi/``：

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --help

参见该目录的 ``README.md``。转换时写入新目录，不覆盖旧 PPO 使用的权重。
只有实现 state_dict/load_state_dict 的 loader 会额外保存数据位置；官方无限迭代
OpenPI loader 不保证逐样本精确续训。

验证
------

.. code-block:: bash

   python -m pytest tests/unit_tests/test_openpi_rlinf_sft.py -q

测试使用缩小的真实 SigLIP / Gemma 网络执行 flow-matching 前后向、参数更新、
权重保存和重新加载，并检查学习率、精度及 horizon 校验。
实际 4090 IsaacLab PPO 验证命令和结果记录在 ``docs/openpi_rlinf_port.md``。
