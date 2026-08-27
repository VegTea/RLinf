Isaac Sim 6 split environments
==============================

Use this setup when OpenPI and Isaac Sim 6 cannot share one Python environment.
Actor and rollout processes run in the Python 3.12 ``.venv-model312`` environment;
Isaac Sim 6 and Isaac Lab 3 environment workers run in the Python 3.12
``.venv-isaacsim6`` environment.
Ray requires the same Python major/minor version across all workers; the two
environments keep separate Torch versions (2.7.1 for the model and 2.10.0 for simulation).

Install from official upstream endpoints without a proxy:

.. code-block:: bash

   bash requirements/install_isaacsim6_split.sh \
     --assets-root /path/to/IsaacAssets6.0

The installer removes proxy variables from every network command and rejects Git
proxy or URL-rewrite configuration. It installs Isaac Sim 6.0.1.0 and PyTorch
2.10 with CUDA 12.8, pins Isaac Lab ``release/3.0.0-beta2``, and applies the
repository-owned local asset-root compatibility patch.

Migrate runtime scenarios separately; embeddings, logs, and caches are excluded:

.. code-block:: bash

   python toolkits/isaaclab/migrate_runtime_assets.py \
     --source /path/to/old/RLinf/rlinf/assets_isaaclab \
     --destination "$PWD/rlinf/assets_isaaclab" \
     --manifest requirements/embodied/isaacsim6_runtime_assets_manifest.json

For Stack Cube only, extract a small local asset root from the combined ZIP:

.. code-block:: bash

   python toolkits/isaaclab/extract_cube_assets.py \
     .downloads/isaacsim6/isaac-sim-assets-complete-6.0.1.zip \
     ../IsaacAssets6.0-minimal

The activation scripts use ``../IsaacAssets6.0-minimal`` by default. It contains
Franka, the colored blocks, and SeattleLabTable; nearest100 ``table_copper.usd``
continues to come from the repository-owned runtime assets.

Activate ``.venv-model312`` before starting Ray. The nearest100 configs use same-node
``model`` and ``isaacsim6`` node groups to select their respective interpreters.
Set ``RLINF_NODE_RANK=0`` before ``ray start --head``.

R595 is outside the fully validated Isaac Sim 6.0.1 driver matrix. Treat headless
Kit, Vulkan/RTX rendering, RGB/depth cameras, and a short nearest100 evaluation as
required acceptance tests rather than assuming compatibility from installation.
