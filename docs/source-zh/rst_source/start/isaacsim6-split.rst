Isaac Sim 6 双环境安装
======================

该安装方式面向无法修改主机驱动、需要同时运行 OpenPI 和 Isaac Sim 6 的节点。
它使用 Python 3.12 的 ``.venv-model312`` 运行 actor/rollout，使用 Python 3.12 的
``.venv-isaacsim6`` 运行 Isaac Sim 6/Isaac Lab 3 环境进程。
两个 venv 必须使用相同的 Python 主次版本，这是 Ray worker 的要求；Torch 版本仍分别
保持模型侧 ``2.7.1`` 和仿真侧 ``2.10.0``。

安装
----

安装脚本只访问 NVIDIA、PyTorch、PyPI 和 GitHub 官方地址。每条下载命令都会清除
``HTTP_PROXY``、``HTTPS_PROXY`` 和 ``ALL_PROXY``；如果 Git 配置了代理或 URL
重写，脚本会停止并显示配置来源。
安装器固定 IsaacLab ``release/3.0.0-beta2``，并自动应用仓库内保存的本地资产根目录
兼容补丁；重复安装时允许复用只包含该补丁的 checkout，其他脏改动仍会被拒绝。

.. code-block:: bash

   cd /path/to/RLinf-IsaacSim6
   bash requirements/install_isaacsim6_split.sh \
     --assets-root /path/to/IsaacAssets6.0

迁移旧仓库的运行场景，不复制 embedding、日志或缓存：

.. code-block:: bash

   python toolkits/isaaclab/migrate_runtime_assets.py \
     --source /path/to/old/RLinf/rlinf/assets_isaaclab \
     --destination "$PWD/rlinf/assets_isaaclab" \
     --manifest requirements/embodied/isaacsim6_runtime_assets_manifest.json

如果只运行 Stack Cube，可从已下载的合并 ZIP 提取约 200 MB 的最小资产：

.. code-block:: bash

   python toolkits/isaaclab/extract_cube_assets.py \
     .downloads/isaacsim6/isaac-sim-assets-complete-6.0.1.zip \
     ../IsaacAssets6.0-minimal

激活脚本会优先使用 ``../IsaacAssets6.0-minimal``。该目录包含 Franka、方块和
默认 SeattleLabTable；nearest100 的 ``table_copper.usd`` 仍从仓库自有资产加载。

运行 nearest100
---------------

.. code-block:: bash

   source .venv-model312/bin/activate
   export RLINF_NODE_RANK=0
   ray start --head
   bash examples/embodiment/eval_embodiment.sh \
     isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100

nearest100 配置通过两个同节点 node group 为模型和仿真进程选择不同解释器。
启动 Ray 前必须设置 ``RLINF_NODE_RANK``，并确保 ``RLINF_MODEL_PYTHON``、
``RLINF_ISAACSIM_PYTHON``、``ISAACSIM_ASSET_ROOT`` 和
``RLINF_SCENARIO_ASSET_ROOT`` 已由激活脚本导出。

兼容性验收
----------

R595 驱动没有包含在 Isaac Sim 6.0.1 的完整官方验证矩阵中。安装完成后仍需运行
headless Kit、官方 Stack Cube、RGB/depth 相机和 nearest100 短程评测；若出现
Hydra、Vulkan 或 RTX renderer 崩溃，应将其视为驱动/Kit 层失败，不应通过修改
RLinf 业务代码绕过。
