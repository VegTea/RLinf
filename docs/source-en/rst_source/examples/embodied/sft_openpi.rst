OpenPI Supervised Fine-Tuning
=============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ vision-language-action models.

Run **full-parameter** or **LoRA** supervised fine-tuning on OpenPI (π₀ / π₀.₅) models
with RLinf. SFT is the first stage before reinforcement learning: the model imitates
high-quality demonstrations so RL can keep optimizing from a strong prior.

Overview
--------

Fine-tune π₀ / π₀.₅ on a LeRobot-format dataset — full-parameter or LoRA — on a single node or a multi-node cluster.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: Methods
      :text-align: center

      Full SFT · LoRA

   .. grid-item-card:: Data
      :text-align: center

      LeRobot format

   .. grid-item-card:: Hardware
      :text-align: center

      1+ nodes · GPUs

| **You'll do:** install OpenPI → prepare a LeRobot dataset → compute norm stats → launch ``run_vla_sft.sh`` → watch the training loss.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a LeRobot-format dataset.

Supported Datasets
~~~~~~~~~~~~~~~~~~~

RLinf supports LeRobot-format datasets, selected via the ``config_name`` field. Built-in formats:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - ``config_name``
     - Dataset / environment
   * - ``pi0_maniskill`` · ``pi05_maniskill``
     - ManiSkill
   * - ``pi0_libero`` · ``pi05_libero``
     - LIBERO
   * - ``pi0_aloha_robotwin``
     - RoboTwin (ALOHA)
   * - ``pi0_realworld``
     - Real-world Franka
   * - ``pi05_droid``
     - DROID Franka (joint-velocity)
   * - ``pi05_metaworld``
     - MetaWorld
   * - ``pi05_calvin``
     - CALVIN

Custom Dataset
~~~~~~~~~~~~~~

You can also train on a custom LeRobot dataset format. Refer to the files below:

1. In ``examples/sft/config/custom_sft_openpi.yaml``, set the data format.

.. code:: yaml

  model:
    openpi:
      config_name: "pi0_custom"

2. In ``rlinf/models/embodiment/openpi/__init__.py``, register the data format ``pi0_custom``.

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

3. In ``rlinf/models/embodiment/openpi/dataconfig/custom_dataconfig.py``, define the custom dataset config.

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

Normalization Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

When you train OpenPI on a newly collected LeRobot dataset, compute dataset
normalization statistics before launching SFT. This is especially important for
a real-world collected dataset.

RLinf provides ``toolkits/lerobot/calculate_norm_stats.py`` to calculate norm_stats for ``state`` and ``actions``. You can use it like:

.. code:: bash

   # Local dataset directory (contains meta/info.json):
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id /path/to/realworld_franka_bin_relocation

   # Or a Hugging Face repo id cached under ~/.cache/huggingface/lerobot by default:
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

.. note::

   - ``--repo-id`` accepts a local dataset path or a LeRobot Hugging Face repo id.
   - Optionally set ``HF_LEROBOT_HOME`` to change the cache parent for repo ids (default: ``~/.cache/huggingface/lerobot``).
   - ``config_name`` must match your custom openpi dataconfig used by training.

The script writes the generated stats under ``<assets_dir>/<exp_name>/<repo_id>/norm_stats.json``.
The OpenPI loader later reads the normalization stats from ``<model_path>/<repo_id>`` at runtime.

A practical tip for stable training is to manually check the normalization statistics for very small standard deviations or narrow q99–q01 ranges. Increasing the standard deviation or widening the q99–q01 gap can help stabilize training, especially in two-stage pipelines that transition from SFT to online training.

Installation
------------

.. include:: _setup_common.rst

**Option 1: Docker image** — image tag ``agentic-rlinf0.3-maniskill_libero``:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
      # Mainland China mirror: docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

   # Inside the container, switch to the OpenPI virtual environment:
   source switch_env openpi

**Option 2: Custom environment** — install bundle ``--env maniskill_libero``:

.. code:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Run It
------

**1. Configuration**

Full examples live in:

- ``examples/sft/config/libero_sft_openpi.yaml``
- ``examples/sft/config/realworld_sft_openpi.yaml``

A generic OpenPI SFT config looks like this:

.. code:: yaml

    cluster:
        num_nodes: 1                 # number of nodes
        component_placement:         # component → GPU mapping
            actor: 0-3

To enable LoRA fine-tuning, set ``actor.model.is_lora: True`` and configure ``actor.model.lora_rank``:

.. code:: yaml

    actor:
        model:
            is_lora: True
            lora_rank: 32

**2. Launch**

Start the Ray cluster, then run the helper script:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openpi

The same script works for generic text SFT; just swap the config file.

Visualization and Results
-------------------------

Monitor the **training loss** to confirm the model is imitating the demonstrations. For
every logged metric, see :doc:`Training metrics <../../reference/metrics>`.

.. code-block:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs

Fine-tuning Franka from official π₀.₅-DROID
----------------------------------------------

``droid_sft_openpi_pi05`` fine-tunes the official ``pi05_droid`` checkpoint on
local DROID-style Franka data. The example is configured for
``/inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force``. It
uses ``exterior_image_1_left``, ``wrist_image_left``, seven-dimensional
``joint_position``, and ``gripper_position``. Absolute joint targets in this
dataset are converted at 15 Hz to the joint-velocity actions expected by the
official checkpoint; gripper commands remain absolute.

First convert the official JAX checkpoint to OpenPI PyTorch and set its path.
Then generate normalization statistics for the converted velocity actions:

.. code:: bash

   export PI05_DROID_MODEL_PATH=/path/to/pi05_droid_pytorch
   export PI05_DROID_NORM_STATS=/path/to/pi05_droid_wipe_board_norm_stats

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_droid \
       --repo-id /inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force \
       --output-dir "$PI05_DROID_NORM_STATS"

   bash examples/sft/run_vla_sft.sh droid_sft_openpi_pi05

Do not use ``pi05_droid_polaris`` for this workflow: it is a joint-position
PolaRiS adapter and is incompatible with the official DROID joint-velocity
checkpoint.

Exporting a π₀.₅-DROID model-input video
------------------------------------------

Use the following tool to export the model image inputs for one episode. From
left to right, every frame contains ``base_0_rgb`` (``exterior_image_1_left``),
``left_wrist_0_rgb`` (``wrist_image_left``), and the all-zero,
official-DROID-masked ``right_wrist_0_rgb``. Images are processed with the
official ``resize_with_pad(224, 224)`` transform.

.. code:: bash

   python toolkits/lerobot/extract_openpi_droid_inputs_video.py \
       --episode-index 0

By default, the video is written to
``outputs/extract_videos/episode_000000_pi05_droid_inputs.mp4``. Use
``--dataset-path``, ``--output-dir``, or ``--fps`` to override the defaults.
