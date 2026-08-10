OpenPI 监督微调
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ 视觉-语言-动作模型。

使用 RLinf 对 OpenPI（π₀ / π₀.₅）模型进行 **全量监督微调（Full-parameter SFT）** 或
**LoRA 微调**。SFT 通常作为进入强化学习前的第一阶段：模型先模仿高质量示例，后续强化学习才能在良好先验上继续优化。

概览
----------------------------------------

在 LeRobot 格式数据集上微调 π₀ / π₀.₅——全量或 LoRA——可在单机或多节点集群上进行。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: 方法
      :text-align: center

      Full SFT · LoRA

   .. grid-item-card:: 数据
      :text-align: center

      LeRobot format

   .. grid-item-card:: 硬件
      :text-align: center

      1+ 节点 · GPU

| **你将完成：** 安装 OpenPI → 准备 LeRobot 数据集 → 计算归一化统计 → 启动 ``run_vla_sft.sh`` → 观察训练损失。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 LeRobot 格式的数据集。

支持的数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 支持 LeRobot 格式的数据集，通过 ``config_name`` 字段指定。内置格式如下：

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - ``config_name``
     - 数据集 / 环境
   * - ``pi0_maniskill`` · ``pi05_maniskill``
     - ManiSkill
   * - ``pi0_libero`` · ``pi05_libero``
     - LIBERO
   * - ``pi0_aloha_robotwin``
     - RoboTwin（ALOHA）
   * - ``pi0_realworld``
     - 真机 Franka
   * - ``pi05_droid``
     - DROID Franka（joint-velocity）
   * - ``pi05_metaworld``
     - MetaWorld
   * - ``pi05_calvin``
     - CALVIN

自定义数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

也可通过自定义 LeRobot 数据集格式来训练特定数据集，具体可参考以下文件：

1. 在 ``examples/sft/config/custom_sft_openpi.yaml`` 中，指定数据格式。

.. code:: yaml

  model:
    openpi:
      config_name: "pi0_custom"

2. 在 ``rlinf/models/embodiment/openpi/__init__.py`` 中，注册数据格式 ``pi0_custom``。

.. code:: python

    TrainConfig(
        name="pi0_custom",
        model=pi0_config.Pi0Config(),
        data=CustomDataConfig(
            repo_id="physical-intelligence/custom_dataset",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=True,  # True for delta action, False for abs_action
            action_train_with_rotation_6d=False,  # User can add extra config in custom dataset
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),

3. 在 ``rlinf/models/embodiment/openpi/dataconfig/custom_dataconfig.py`` 中，定义自定义数据集的配置。

.. code:: python

    class CustomDataConfig(DataConfig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.repo_id = "physical-intelligence/custom_dataset"
            self.base_config = DataConfig(
                prompt_from_task=True
            )
            self.assets = AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets")
            self.extra_delta_transform = True
            self.action_train_with_rotation_6d = False

归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当你在新采集的 LeRobot 数据集上训练 OpenPI 时，需要在启动 SFT 之前先计算
归一化统计。这对真实机器人采集的数据集尤其重要。

RLinf 提供了 ``toolkits/lerobot/calculate_norm_stats.py``，用于为
``state`` 和 ``actions`` 计算 ``norm_stats``。使用方式如下：

.. code:: bash

   # 本地数据集目录（包含 meta/info.json）：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id /path/to/realworld_franka_bin_relocation

   # 或使用默认缓存在 ~/.cache/huggingface/lerobot 下的 Hugging Face repo id：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

.. note::

   - ``--repo-id`` 可以是本地数据集路径，也可以是 LeRobot 的 Hugging Face repo id。
   - 可选：通过 ``HF_LEROBOT_HOME`` 修改 repo id 的缓存父目录（默认：``~/.cache/huggingface/lerobot``）。
   - ``config_name`` 必须与训练时使用的自定义 OpenPI dataconfig 一致。

该脚本会将生成的统计信息写入 ``<assets_dir>/<exp_name>/<repo_id>/norm_stats.json``。
OpenPI 加载器会在运行时从 ``<model_path>/<repo_id>`` 读取归一化统计信息。

另一个有助于稳定训练的实用建议是，手动检查归一化统计中是否存在非常小的标准差，
或过窄的 q99-q01 区间。适当增大标准差，或拉宽 q99-q01 的范围，通常有助于提升
训练稳定性，尤其是在先做 SFT 再进入在线训练的两阶段流程中。

安装
----------------------------------------

.. include:: _setup_common.rst

**方式一：使用 Docker 镜像** —— 镜像标签 ``agentic-rlinf0.3-maniskill_libero``：

.. code:: bash

    docker run -it --rm --gpus all \
        --shm-size 20g \
        --network host \
        --name rlinf \
        -v .:/workspace/RLinf \
        rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
        # 国内镜像加速：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

    # 进入容器后，切换到 OpenPI 虚拟环境：
    source switch_env openpi

**方式二：自建环境** —— 安装套件 ``--env maniskill_libero``：

.. code:: bash

    # 为提高国内依赖安装速度，可以添加 --use-mirror。
    bash requirements/install.sh embodied --model openpi --env maniskill_libero
    source .venv/bin/activate

运行
----------------------------------------

**1. 配置**

完整示例配置位于：

- ``examples/sft/config/libero_sft_openpi.yaml``
- ``examples/sft/config/realworld_sft_openpi.yaml``

通用的 OpenPI SFT 配置示例如下：

.. code:: yaml

    cluster:
        num_nodes: 1                 # 节点数
        component_placement:         # 组件 → GPU 映射
            actor: 0-3

若需要 LoRA 微调，将 ``actor.model.is_lora`` 设为 ``True``，并配置 ``actor.model.lora_rank``：

.. code:: yaml

    actor:
        model:
            is_lora: True
            lora_rank: 32

**2. 启动**

先启动 Ray 集群，再执行训练脚本：

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openpi

同一脚本也适用于通用文本 SFT，只需替换配置文件即可。

可视化与结果
----------------------------------------

关注 **训练损失** 即可确认模型是否在拟合示例数据。各项指标的含义见
:doc:`训练指标 <../../reference/metrics>`。

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs

使用官方 π₀.₅-DROID 微调 Franka
----------------------------------------

``droid_sft_openpi_pi05`` 用于将官方 ``pi05_droid`` 检查点微调到本地 DROID
风格 Franka 数据。示例数据路径已配置为
``/inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force``。它使用
右侧外部视角 ``exterior_image_2_left``、``wrist_image_left``、7 维
``joint_position`` 和 ``gripper_position``；模型第三路图像以全零填充并掩蔽。
数据集中的绝对关节目标会按 15 Hz 转为官方检查点所需的 joint-velocity 动作；
夹爪命令保持绝对值。

默认使用 seed 0 随机留出 20 条完整轨迹，并使用其余 176 条训练，因此测试轨迹中的帧
不会进入训练 loader。对于当前数据集，这对应 130,650 个训练帧和 14,361 个测试帧。
RLinf 每隔 1,000 个 optimizer step，从每条测试轨迹解码 10 个固定噪声 action chunk
（共 200 个），并在 ``eval/`` 下记录归一化 action、joint velocity、gripper position
及积分后 joint position 的 MSE/MAE。具体 episode ID 和 chunk offset 会写入实验目录下
的 ``episode_split.json``。

仓库中的 expert-only FSDP 配置已在双 RTX 4090 上完成 3 个 optimizer step 的
smoke test。配置让冻结的视觉语言塔在各卡复制，并手工包装 OpenPI 的复合 expert
模块，因为 OpenPI 会直接执行内部 Gemma layer 的组件。

仓库中的示例默认使用下面的本地路径；只有 checkpoint 或统计量位于其他位置时才需
覆盖前两个变量。将 ``OPENPI_DATA_HOME`` 指向本地 tokenizer cache，然后启动 SFT：

.. code:: bash

   export PI05_DROID_MODEL_PATH=/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch
   export PI05_DROID_NORM_STATS="$PI05_DROID_MODEL_PATH/assets/wipe_board_v1_zed196_force"
   export OPENPI_DATA_HOME=/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/openpi_cache

   CUDA_VISIBLE_DEVICES=0,1 bash examples/sft/run_vla_sft.sh droid_sft_openpi_pi05

在 4 张 H100 上进行正式的 expert-only 训练时，使用下面经过检查的入口。默认配置为
每卡 micro batch 16、global batch 64、峰值学习率 ``5e-5``、500 个 warmup
step、cosine 衰减和 10,000 个 optimizer step。脚本还会每秒将 GPU 显存和利用率
采样到 ``gpu_metrics.csv``。global batch 为 64 时共处理约 640,000 个样本，相当于
对 130,650 帧训练划分训练约 4.9 遍。

.. code:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash examples/sft/pi05_droid_wipe_board/run_h100_train.sh

每次运行会写入 ``outputs/pi05_droid_h100_train/`` 下带时间戳的目录。这组默认值是
根据双 4090 sweep 得到的安全起点；最终 checkpoint 应结合解码 action 指标和真机
成功率选择，在增大 H100 batch 前也应重新运行 batch probe。

与 JAX 对齐的 RLinf PyTorch 实现
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

上面的命令使用 OpenPI 官方 PyTorch wrapper（``model_type: openpi``）。如需与 OpenPI
JAX 对比，应使用 RLinf 自包含、与 JAX 对齐的 ``droid_sft_openpi_pytorch_pi05``
（``model_type: openpi_pytorch``）。首先用 ``jax2new`` 转换器转换官方 JAX checkpoint；
输出目录必须包含 ``model.safetensors`` 和擦白板归一化资产。

对齐配置使用 FP32 optimizer 主权重、BF16 FSDP 计算、FP32 reduce，并启用非重入式
gradient checkpointing。SigLIP、Gemma expert 0 和共享 embedding 会被冻结，只有
Gemma action expert 1 以及 action/time projection 参与训练。它与官方 wrapper baseline
使用相同的 176/20 轨迹划分及解码评估。

.. code:: bash

   export PI05_DROID_RLINF_MODEL_PATH=/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch_rlinf
   export PI05_DROID_RLINF_NORM_STATS="$PI05_DROID_RLINF_MODEL_PATH/assets/wipe_board_v1_zed196_force"

   # 一个更新 step，并解码评估 20 个 chunk。
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash examples/sft/pi05_droid_wipe_board/run_openpi_pytorch_h100_smoke.sh

   # 训练 10,000 step；每 1,000 step 解码评估 200 个 chunk。
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash examples/sft/pi05_droid_wipe_board/run_openpi_pytorch_h100_train.sh

对齐版本输出到 ``outputs/pi05_droid_openpi_pytorch_h100/``。不要把旧版官方 wrapper 的
``model.safetensors`` 直接传给该配置：两者参数布局不同，strict load 会明确拒绝。

所有训练结束后，用下面的命令生成三方解码指标对比，避免混淆两个 PyTorch 实现：

.. code:: bash

   python examples/sft/pi05_droid_wipe_board/compare_sft_evaluations.py \
       --jax-metrics /path/to/openpi/eval_metrics.json \
       --aligned-rlinf-run /path/to/openpi_pytorch_run \
       --wrapper-baseline-run /path/to/openpi_wrapper_run \
       --output /path/to/final_eval_comparison.json

该工具会把 RLinf TensorBoard step ``N - 1`` 映射回 checkpoint ``N``，检查每条解码
指标是否完整且为有限值，并在结果中分别标识对齐实现和官方 wrapper baseline。

擦白板数据的归一化统计已生成到上述路径。如需重新生成，使用无需解码图像的数值列
快速路径；它会保留官方 15-step action chunk 和 episode 尾部 padding 语义，同时
跳过 parquet 内嵌的三路相机图像：

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_droid \
       --repo-id /inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force \
       --output-dir "$PI05_DROID_NORM_STATS" \
       --numeric-only

该配置不使用 ``pi05_droid_polaris``，后者是 joint-position / PolaRiS 适配，不能与
官方 DROID joint-velocity 检查点混用。

导出 π₀.₅-DROID 模型图像输入视频
------------------------------------------

使用以下工具导出单个 episode 的模型图像输入视频。每一帧从左到右依次为
``base_0_rgb``（选中的外部相机）、``left_wrist_0_rgb``
（``wrist_image_left``）和官方 DROID policy 掩蔽的全零 ``right_wrist_0_rgb``。
默认选择右侧外部视角 ``exterior_image_2_left``。图像会按官方
``resize_with_pad(224, 224)`` 处理。

.. code:: bash

   python toolkits/lerobot/extract_openpi_droid_inputs_video.py \
       --episode-index 0

视频默认保存至 ``outputs/extract_videos/episode_000000_pi05_droid_inputs.mp4``。
用 ``--external-camera left`` 可改选 ``exterior_image_1_left``；用
``--dataset-path``、``--output-dir`` 和 ``--fps`` 可覆盖其他默认值。

以绝对关节目标部署 π₀.₅-DROID
------------------------------------------

下面的部署入口使用 OpenPI 的 ``WebsocketPolicyServer``。它保持擦白板训练的输入
映射（右侧外部相机、腕部相机、掩蔽的全零第三路），并在模型动作反归一化后将 7 维
joint-velocity chunk 转成绝对关节目标。控制频率 ``f=15 Hz`` 时，计算方式为
``q_target[k] = q_current + cumsum(dq[0:k]) / f``；第 8 维绝对夹爪命令保持不变。

在 GPU 机器启动 policy server。所有路径都必须是本地路径；命令会显式清除代理
变量，因此不会通过代理下载模型或数据：

.. code:: bash

   unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
   export OPENPI_DATA_HOME=/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/openpi_cache

   python toolkits/standalone_eval_scripts/openpi/serve_pi05_droid.py \
       --checkpoint-dir /inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch \
       --norm-stats-dir /inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch/assets/wipe_board_v1_zed196_force \
       --control-frequency-hz 15 \
       --pytorch-device cuda \
       --host 0.0.0.0 \
       --port 8000

必须使用与部署 checkpoint 对应的归一化统计。上述路径对应擦白板 SFT 数据；未经
微调的官方 checkpoint 应改用 ``assets/droid``。

DROID 真机进程可以使用 OpenPI 官方客户端。每次请求必须携带当前机器人状态；每次
响应包含一个 ``(15, 8)`` 的绝对动作 chunk：

.. code:: python

   from openpi_client.websocket_client_policy import WebsocketClientPolicy

   policy = WebsocketClientPolicy(host="GPU_SERVER_IP", port=8000)
   result = policy.infer(
       {
           "observation/exterior_image_2_left": right_image_uint8_hwc,
           "observation/wrist_image_left": wrist_image_uint8_hwc,
           "observation/joint_position": joint_position_float32_7,
           "observation/gripper_position": gripper_position_float32_1,
           "prompt": "wipe the whiteboard",
       }
   )
   absolute_action_chunk = result["actions"]

真机端应以 15 Hz 依次发送 chunk 中的目标位置。该转换不能替代真机端的关节限位、
单步运动限幅、watchdog 和急停；启用机械臂运动前必须验证这些保护措施。
