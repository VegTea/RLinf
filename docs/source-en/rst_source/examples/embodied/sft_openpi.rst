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
uses the right exterior view ``exterior_image_2_left``, ``wrist_image_left``,
seven-dimensional ``joint_position``, and ``gripper_position``. The third model
image slot is zero-filled and masked. Absolute joint targets in this dataset are
converted at 15 Hz to the joint-velocity actions expected by the official
checkpoint; gripper commands remain absolute.

By default, seed 0 holds out 20 complete trajectories and trains on the other
176, so frames from a test trajectory never enter the training loader. This is
130,650 training frames and 14,361 test frames for the current dataset. Every
1,000 optimizer steps, RLinf decodes 10 fixed-noise action chunks from each
held-out trajectory (200 chunks total) and logs normalized-action, joint-
velocity, gripper-position, and integrated joint-position MSE/MAE under
``eval/``. The exact episode IDs and chunk offsets are saved in
``episode_split.json`` under the experiment directory.

The checked-in expert-only FSDP configuration was smoke-tested for three
optimizer steps on two RTX 4090 GPUs. It keeps the frozen vision-language tower
replicated and manually wraps the composite OpenPI expert module because OpenPI
executes the inner Gemma layer components directly.

The checked-in example defaults to the local paths below. Override the first
two variables only when the checkpoint or statistics live elsewhere. Set
``OPENPI_DATA_HOME`` to the local tokenizer cache, then launch SFT:

.. code:: bash

   export PI05_DROID_MODEL_PATH=/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch
   export PI05_DROID_NORM_STATS="$PI05_DROID_MODEL_PATH/assets/wipe_board_v1_zed196_force"
   export OPENPI_DATA_HOME=/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/openpi_cache

   CUDA_VISIBLE_DEVICES=0,1 bash examples/sft/run_vla_sft.sh droid_sft_openpi_pi05

For a production expert-only run on four H100 GPUs, use the checked entry point
below. It defaults to micro batch 16 per GPU, global batch 64, peak learning
rate ``5e-5``, 500 warmup steps, cosine decay, and 10,000 optimizer steps. It
also samples GPU memory and utilization once per second into ``gpu_metrics.csv``.
At global batch 64, this processes about 640,000 samples, or 4.9 passes over
the 130,650-frame training split.

.. code:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash examples/sft/pi05_droid_wipe_board/run_h100_train.sh

The script writes each run to a timestamped directory under
``outputs/pi05_droid_h100_train/``. These defaults are a safe starting point
from a dual-4090 sweep; select the final checkpoint with decoded action metrics
and robot success rate, and re-run a batch probe before increasing the H100
batch.

JAX-aligned RLinf PyTorch implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command above uses the official OpenPI PyTorch wrapper (``model_type:
openpi``). To compare against OpenPI JAX with RLinf's self-contained,
JAX-aligned implementation, use ``droid_sft_openpi_pytorch_pi05``
(``model_type: openpi_pytorch``). Convert the official JAX checkpoint with the
``jax2new`` converter first; the resulting directory must contain
``model.safetensors`` and the wipe-board normalization assets.

The aligned recipe keeps FP32 optimizer master weights, uses BF16 FSDP compute
and FP32 reductions, enables non-reentrant gradient checkpointing, and freezes
SigLIP, Gemma expert 0, and the shared embedding. Gemma action expert 1 and the
action/time projections remain trainable. It uses the same 176/20 trajectory
split and decoded evaluation as the official-wrapper baseline.

.. code:: bash

   export PI05_DROID_RLINF_MODEL_PATH=/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch_rlinf
   export PI05_DROID_RLINF_NORM_STATS="$PI05_DROID_RLINF_MODEL_PATH/assets/wipe_board_v1_zed196_force"

   # One update and one 20-chunk decoded evaluation.
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash examples/sft/pi05_droid_wipe_board/run_openpi_pytorch_h100_smoke.sh

   # 10,000 updates and 200-chunk evaluation every 1,000 updates.
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash examples/sft/pi05_droid_wipe_board/run_openpi_pytorch_h100_train.sh

Aligned runs are written below ``outputs/pi05_droid_openpi_pytorch_h100/``.
Do not point this configuration at the older official-wrapper
``model.safetensors``: its parameter layout differs from the ``jax2new``
checkpoint and strict loading will reject it.

Once all runs finish, generate a three-way decoded-metric comparison without
mixing up the two PyTorch implementations:

.. code:: bash

   python examples/sft/pi05_droid_wipe_board/compare_sft_evaluations.py \
       --jax-metrics /path/to/openpi/eval_metrics.json \
       --aligned-rlinf-run /path/to/openpi_pytorch_run \
       --wrapper-baseline-run /path/to/openpi_wrapper_run \
       --output /path/to/final_eval_comparison.json

The tool maps RLinf TensorBoard step ``N - 1`` back to checkpoint ``N``, checks
that every decoded metric row is complete and finite, and keeps the aligned
implementation and official-wrapper baseline under distinct result names.

The wipe-board normalization statistics have already been generated at the
path above. To regenerate them, use the image-free numeric path; it preserves
the official 15-step action-chunk and episode-end padding semantics without
decoding the three parquet-embedded camera streams:

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_droid \
       --repo-id /inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force \
       --output-dir "$PI05_DROID_NORM_STATS" \
       --numeric-only

Do not use ``pi05_droid_polaris`` for this workflow: it is a joint-position
PolaRiS adapter and is incompatible with the official DROID joint-velocity
checkpoint.

Exporting a π₀.₅-DROID model-input video
------------------------------------------

Use the following tool to export the model image inputs for one episode. From
left to right, every frame contains ``base_0_rgb`` (the selected exterior
camera), ``left_wrist_0_rgb`` (``wrist_image_left``), and the all-zero,
official-DROID-masked ``right_wrist_0_rgb``. The right exterior view
(``exterior_image_2_left``) is selected by default. Images are processed with
the official ``resize_with_pad(224, 224)`` transform.

.. code:: bash

   python toolkits/lerobot/extract_openpi_droid_inputs_video.py \
       --episode-index 0

By default, the video is written to
``outputs/extract_videos/episode_000000_pi05_droid_inputs.mp4``. Use
``--external-camera left`` to select ``exterior_image_1_left`` instead. Use
``--dataset-path``, ``--output-dir``, or ``--fps`` to override other defaults.

Serving π₀.₅-DROID with absolute joint targets
------------------------------------------------

The deployment entry point below uses OpenPI's ``WebsocketPolicyServer``. It
keeps the wipe-board input mapping (right exterior camera, wrist camera, and a
masked zero third slot), then converts the unnormalized seven-dimensional
joint-velocity chunk into absolute joint targets. For a control frequency
``f=15 Hz``, it computes
``q_target[k] = q_current + cumsum(dq[0:k]) / f``. The eighth, absolute gripper
command is passed through unchanged.

Start the policy server on the GPU machine. All paths must be local; the
command explicitly removes proxy variables and therefore performs no proxied
model or data download:

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

Use the normalization statistics associated with the deployed checkpoint. The
path above is for the wipe-board SFT data. The untouched official checkpoint
instead uses ``assets/droid``.

The DROID robot process can use OpenPI's official client. Each request must
contain the current robot state; each response contains a ``(15, 8)`` absolute
action chunk:

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

Send the chunk's waypoints to the robot sequentially at 15 Hz. The conversion
does not replace robot-side joint limits, per-step motion limits, watchdogs, or
emergency-stop handling. Validate those protections before enabling motion.
