OpenPI_RLinf SFT (coexistence port)
==================================

This branch keeps the legacy ``model_type: openpi`` used by IsaacLab PPO and adds
``model_type: openpi_rlinf`` for the upstream JAX-aligned PyTorch Pi0/Pi0.5 implementation.
Local smoke tests do not independently establish full numerical parity with JAX.

Provenance
----------

The baseline is RLinf/RLinf ``b023deb7e90ceadfe18e0a528e862e3c630f7e0a``.
SFT loader, normalization, horizon validation and compile-toggle updates from
``184ba07f`` are included without deleting or renaming the legacy implementation.
Data configurations and policy transforms are isolated under
``rlinf/models/embodiment/openpi_rlinf/``. The port does not migrate unrelated
scheduler/model refactors. Task subclasses are included as upstream source, but
new-model integration is scoped to SFT; existing PPO continues using legacy OpenPI.
DINOv2 and a random-initialization training entry point are not implemented here.

Usage
-----

Use a separate model environment with the existing RLinf/OpenPI dependencies.
The neural network is PyTorch, but official dataset transforms still depend on
OpenPI/JAX, LeRobot, tokenizer and dataset assets. BEHAVIOR requires its additional
video/dataset dependencies. Do not upgrade a shared PPO environment in place.

Available recipes:

- ``examples/sft/config/robotwin_sft_openpi_rlinf.yaml`` (Pi0).
- ``examples/sft/config/behavior_sft_openpi_pi05_rlinf.yaml`` (Pi0.5).
- ``examples/sft/config/realworld_sft_openpi_pytorch_dual_franka_tcp_rot6d.yaml``
  (the historical filename is retained; the model type is ``openpi_rlinf``).

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_rlinf \
     actor.model.model_path=/absolute/path/to/new_format_model \
     data.train_data_paths=/absolute/path/to/lerobot_dataset \
     actor.model.openpi_data.norm_stats_path=/absolute/path/to/norm_stats.json

Unlike the legacy entry point, train_data_paths denotes a dataset root or repo id,
not a cache root. Explicit normalization paths must exist. If omitted, the loader
warns and uses the TrainConfig defaults. Register new data configurations under
``rlinf.models.embodiment.openpi_rlinf.dataconfig``.

Constraints
-----------

- FSDP1 with ``sharding_strategy: no_shard`` and ``use_orig_params: true``.
- Model ``precision: null`` preserves the upstream bf16/fp32 parameter islands;
  all FSDP mixed-precision dtypes must be null. An all-fp32 configuration is also
  accepted. FSDP2 and nested full sharding are not supported by this port.
- ``openpi_cosine`` reproduces the upstream nonzero initial warmup learning rate.
- ``+actor.model.openpi.torch_compile=false`` disables fused-kernel compilation.
- The official SFT loader uses TrainConfig's action horizon. Execution chunks may
  be shorter; an explicit incompatible model horizon is rejected.
- New-model max_epochs counts dataset epochs; max_steps caps optimizer steps.
  Legacy SFT/PPO counting remains unchanged.
- Upstream ``Pi0.sft_forward`` currently disables gradient checkpointing, regardless
  of the YAML switch. Measure memory before launching large jobs.

Weights and tests
-----------------

The factory requires ``model.safetensors`` or ``full_weights.pt``; if both exist,
full_weights wins with a warning. Use ``runner.resume_dir`` to restore optimizer
and scheduler, rather than merely pointing model_path at weights. Loader positions
are saved only for stateful loaders; the official infinite OpenPI loader does not
provide exact sample-order resume.

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --help
   python -m pytest tests/unit_tests/test_openpi_rlinf_sft.py -q

See ``rlinf/utils/ckpt_convertor/openpi/README.md`` for conversion modes. Write
converted weights to a new directory, never over existing PPO checkpoints.
The CPU tests exercise real, dimension-reduced SigLIP/Gemma flow-matching training,
weight round trips, the schedule and validation guards. The 4090 IsaacLab PPO
runtime evidence is recorded in ``docs/openpi_rlinf_port.md``.
