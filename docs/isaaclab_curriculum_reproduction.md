# RLinf IsaacLab Curriculum Reproduction Notes

This document records the sidecar work added to RLinf/IsaacLab for table/cube/camera scenario reset, curriculum training, diagnostics, rendering, evaluation, and GPU smoke tests.

The goal is to let another agent reproduce the current behavior without reading the full conversation history.

## Scope

The work targets the IsaacLab Franka stack-cube task with OpenPI pi0.5:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05*.yaml
```

The important behavior change is that training/evaluation can reset each IsaacLab env from a JSONL scenario file instead of only using the original random cube reset. A scenario can parameterize:

- table USD asset
- cube positions/orientations
- table camera pose
- scenario id and distance metadata

Curriculum training then controls which scenario ids are allowed at each stage.

## Important Directories

Repository root:

```text
/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf
```

Scenario and curriculum assets:

```text
rlinf/assets_isaaclab/all_setting/
```

Colored SeattleLabTable USD assets:

```text
rlinf/assets_isaaclab/SeattleLabTable/color_tables/
```

Logs:

```text
logs/
```

## Core RLinf Code Changes

### Scenario Loading

New file:

```text
rlinf/envs/isaaclab/scenario_loader.py
```

Responsibilities:

- Load JSONL scenario records by id.
- Resolve `table_asset` into an actual USD path.
- Default table asset root is:

```text
rlinf/assets_isaaclab/SeattleLabTable/color_tables/
```

Resolution logic:

- If `table_asset` is absolute, use it directly.
- If relative, join it with the table asset root.
- If extension is `.usd`, also try `.usda`; if `.usda`, also try `.usd`.
- If no extension, try both `.usd` and `.usda`.

Important portability note:

- The table USD files may contain texture paths. Check whether those paths are absolute before moving to a new machine.
- The table USD path itself is resolved by RLinf, but texture references inside USD are resolved by USD/IsaacSim. If the USD contains absolute PNG paths, those must exist on the target machine or be rewritten.

### Scenario Scheduling

New file:

```text
rlinf/envs/isaaclab/scenario_scheduler.py
```

Supported reset modes:

- `sequential`: cycle through scenario ids in file order.
- `random`: sample scenario ids randomly.
- `by_id`: use `fixed_ids`.
- `external`: sample from group A/B ids by ratio.

Distributed behavior:

- The scheduler receives `worker_rank`, `total_workers`, and `envs_per_worker` through the IsaacLab event params.
- For random mode, the seed is offset by batch index, so different workers/cards do not keep receiving exactly the same scenario batch.
- For curriculum mode, each worker samples only from the current allowed scenario ids.

Curriculum behavior:

- Can use threshold-based selection from `distance_key`.
- Can use `stage_manifest_file`; when present, exact `cumulative_ids` from the manifest define each stage.
- `incremental_ids` in the manifest are used to separate old/new groups for promotion diagnostics.
- Changing curriculum stage does not rebuild the IsaacLab env; it only updates the scheduler. The next reset applies the new allowed id set.

### IsaacLab Env Wrapper

Modified:

```text
rlinf/envs/isaaclab/tasks/stack_cube.py
rlinf/envs/isaaclab/isaaclab_env.py
rlinf/envs/isaaclab/venv.py
```

Key changes:

- `stack_cube.py` reads `cfg.init_params.scenario_reset`.
- If enabled, it replaces the original cube randomization event with `franka_stack_events.apply_scenario_reset`.
- It passes scenario file path, reset mode, fixed ids, external sampling ids, worker rank, total workers, envs per worker, seed, and curriculum config to the event.
- It disables table material randomization when scenario reset owns table assets.
- It keeps `grid_reset` and `cube_pose_random_reset` as fallback paths when `scenario_reset.enabled=false`.

`isaaclab_env.py` now:

- Tracks the active scenario record per env.
- Emits `scenario_id` in episode metrics.
- Supports reset snapshots: table camera PNG, optional wrist PNG, and metadata JSONL.
- For trajectory/eval recording, stores scenario metadata with each episode.
- Exposes:

```text
set_scenario_curriculum_stage(stage_index)
set_scenario_curriculum_progress(stage_index, stage_step)
get_scenario_curriculum_state()
get_trajectory_record_counts()
```

`venv.py` now:

- Propagates `scenario_records` returned by IsaacLab reset back to RLinf.
- Supports replay/visual helper commands used by rendering and trajectory tools.
- Detects IsaacLab subprocess exit while waiting for reset/step and returns a real failure instead of hanging forever.

### Env Worker Diagnostics

Modified:

```text
rlinf/workers/env/env_worker.py
```

Key changes:

- Supports per-worker log files under:

```text
<run_log_dir>/worker_logs/
```

- Writes curriculum scenario stats when enabled:

```text
worker_logs/curriculum_scenario_stats_rank_<rank>.jsonl
```

Each row includes per-scenario sample counts and success rate for that worker/local step.

- Adds Ray actor methods used by the runner:

```text
set_scenario_curriculum_stage(stage_index)
set_scenario_curriculum_progress(stage_index, stage_step)
get_eval_trajectory_record_counts()
```

### Runner Curriculum Logic

Modified:

```text
rlinf/runners/embodied_runner.py
```

Added scenario curriculum management independent of the older embedding-buffer curriculum.

Supported promotion modes:

1. Legacy hit-count mode:

```yaml
success_threshold: 0.8
success_thresholds: [...]
stable_steps: 3
require_consecutive_success: false
```

Meaning:

- At the current stage, count training updates where `env/success_once >= threshold`.
- If `require_consecutive_success=false`, hits do not need to be consecutive.
- Advance after `stable_steps` hits.

2. Grouped moving-average mode:

```yaml
promotion:
  mode: grouped_moving_average
  window: 5
  min_steps_per_stage: 20
  cumulative_success_thresholds: [...]
  new_success_thresholds: [...]
  cumulative_floor: [...]
```

Meaning:

- Stage 0 checks cumulative success only.
- Stage >0 checks the newly added scenarios separately with `new_success_thresholds`.
- It also checks a floor on all currently allowed scenarios via `cumulative_floor`.
- This was added to avoid easy old settings inflating the average and promoting too early.

Optional forced promotion:

```yaml
promotion:
  max_steps_per_stage: 300
```

Meaning:

- If configured, force advance after 300 curriculum stage steps even if metrics are below threshold.
- The original YAMLs do not include this; separate `_max300` YAMLs were created for A/B tests.
- Promotion logs include `reason=metrics` or `reason=max_steps_per_stage`.

Runner logs/TensorBoard include:

```text
curriculum/stage_index
curriculum/threshold
curriculum/allowed_scenarios
curriculum/stage_steps
curriculum/success_hits
curriculum/success_threshold
curriculum/success_window_mean
curriculum/new_success_window_mean
curriculum/old_success_window_mean
curriculum/cumulative_success_window_mean
curriculum/new_success_threshold
curriculum/cumulative_floor
curriculum/min_steps_per_stage
curriculum/max_steps_per_stage
curriculum/mix_old_ratio
curriculum/mix_new_ratio
curriculum/advanced
```

## IsaacLab External Patch Point

RLinf imports this module at runtime:

```python
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
```

This work assumes that module provides at least:

```python
apply_scenario_reset(...)
noop_event(...)
grid_traverse_object_pose(...)
randomize_object_pose(...)
```

The important custom function is `apply_scenario_reset`. It must:

- Load the JSONL scenario file with `ScenarioLoader`.
- Use `ScenarioScheduler` to select per-env scenario ids.
- Apply cube poses from each scenario.
- Replace the table prim reference with the selected `table_asset` USD.
- Apply `table_cam_pos` and `table_cam_rot` if present.
- Save the last record per env, usually as `_scenario_last_record_by_env`, so `venv.py` and `isaaclab_env.py` can propagate `scenario_records`.
- Avoid rebuilding the whole IsaacLab app/env on every scenario. The fast path should swap/reset in the existing stage and take roughly seconds, not one IsaacLab launch per scenario.

The current git tree does not show this file as a tracked RLinf file. When moving to another machine, confirm the IsaacLab package/extension in that environment contains the same functions.

## Scenario Data

Generated/used scenario files live under:

```text
rlinf/assets_isaaclab/all_setting/
```

Important files:

```text
color_tables.jsonl
combined_scenarios/Isaaclab_all_scenarios_10025.jsonl
combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl
embedding/metadata.csv
embedding/id_to_row.json
embedding/embeddings_l2norm.npy
```

`color_tables.jsonl` was made from all USD file names under:

```text
rlinf/assets_isaaclab/SeattleLabTable/color_tables/
```

The 10025 scenario file combines table/cube/camera variants. Distance files add cosine distance to reference setting `009876`.

Generation helpers:

```text
examples/embodiment/generate_all_setting_combinations.py
examples/embodiment/generate_table_cube_pose_scenarios.py
examples/embodiment/generate_table_cam_circle_poses.py
examples/embodiment/generate_curriculum_stage_manifest.py
examples/embodiment/generate_dynamic_curriculum_stage_manifest.py
```

## Curriculum Data

### Static 10-stage / 200-scenario curriculum

Directory:

```text
rlinf/assets_isaaclab/all_setting/curriculum_10stage_200/
```

Main files:

```text
Isaaclab_10stage_200_with_distance.jsonl
Isaaclab_10stage_200_stage_manifest.json
stage_sets/incremental/stage_00.jsonl ...
stage_sets/cumulative/stage_00.jsonl ...
```

Cumulative counts:

```text
10,20,40,60,80,100,120,140,170,200
```

### Dynamic top-10 10-stage / 200-scenario curriculum

Directory:

```text
rlinf/assets_isaaclab/all_setting/curriculum_10stage_200_dynamic_top10/
```

Generated by:

```bash
.venv/bin/python examples/embodiment/generate_dynamic_curriculum_stage_manifest.py \
  --scenario-file rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl \
  --metadata-csv rlinf/assets_isaaclab/all_setting/embedding/metadata.csv \
  --embeddings rlinf/assets_isaaclab/all_setting/embedding/embeddings_l2norm.npy \
  --output-dir rlinf/assets_isaaclab/all_setting/curriculum_10stage_200_dynamic_top10 \
  --counts 10,20,40,60,80,100,120,140,170,200 \
  --prefix Isaaclab_10stage_200_dynamic_top10 \
  --reference-id 009876 \
  --top-k 10
```

### Dynamic top-10 19-stage / 10025-scenario curriculum

Directory:

```text
rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_dynamic_top10/
```

Generated by:

```bash
.venv/bin/python examples/embodiment/generate_dynamic_curriculum_stage_manifest.py \
  --scenario-file rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl \
  --metadata-csv rlinf/assets_isaaclab/all_setting/embedding/metadata.csv \
  --embeddings rlinf/assets_isaaclab/all_setting/embedding/embeddings_l2norm.npy \
  --output-dir rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_dynamic_top10 \
  --counts 50,100,200,400,600,800,989,1200,1500,1997,2500,3000,4048,5000,5964,7000,8014,9000,10025 \
  --prefix Isaaclab_19stage_10025_dynamic_top10 \
  --reference-id 009876 \
  --top-k 10
```

Dynamic top-10 method:

- Stage 0 is selected by cosine distance to reference scenario `009876`.
- Later stages treat all previously selected scenarios as the mastered set.
- For each unselected scenario, compute the mean cosine distance to the top-10 nearest mastered embeddings.
- Add the nearest candidates until the target cumulative count is reached.

Output record fields include:

```text
selected_stage
reference_cosine_distance
dynamic_top10_mean_cosine_distance
nearest_mastered_ids
nearest_mastered_distances
```

Additional note:

```text
rlinf/assets_isaaclab/all_setting/dynamic_top10_curriculum_README.md
```

### Structured parameter-distance 19-stage / 10025-scenario curriculum

Directory:

```text
rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_structured/
```

The structured distance is model-free and uses a single reference scenario, matching the original non-top-k curriculum style. It compares each scenario to reference scenario `009876` using normalized weighted components from the JSONL parameters:

```text
structured_distance =
  0.35 * cube position distance
+ 0.25 * cube relative-layout distance
+ 0.05 * cube rotation distance
+ 0.20 * camera position distance
+ 0.10 * camera rotation distance
+ 0.05 * table asset changed
```

Generated by:

```bash
.venv/bin/python examples/embodiment/generate_structured_scenario_distance.py \
  --source rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl \
  --output rlinf/assets_isaaclab/all_setting/combined_with_structured_distance/Isaaclab_all_scenarios_10025_with_structured_distance.jsonl \
  --summary rlinf/assets_isaaclab/all_setting/combined_with_structured_distance/structured_distance_summary.json \
  --reference-id 009876

.venv/bin/python examples/embodiment/generate_curriculum_stage_manifest.py \
  --source rlinf/assets_isaaclab/all_setting/combined_with_structured_distance/Isaaclab_all_scenarios_10025_with_structured_distance.jsonl \
  --output-dir rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_structured \
  --distance-key structured_distance \
  --counts 50,100,200,400,600,800,989,1200,1500,1997,2500,3000,4048,5000,5964,7000,8014,9000,10025 \
  --prefix Isaaclab_19stage_10025_structured
```

Main files:

```text
combined_with_structured_distance/Isaaclab_all_scenarios_10025_with_structured_distance.jsonl
combined_with_structured_distance/structured_distance_summary.json
curriculum_19stage_10025_structured/Isaaclab_19stage_10025_structured_with_distance.jsonl
curriculum_19stage_10025_structured/Isaaclab_19stage_10025_structured_stage_manifest.json
```

Evaluate the structured 19-stage ranges:

```bash
bash examples/embodiment/eval_structured_curriculum_ranges.sh
```

Useful smoke-test command:

```bash
bash examples/embodiment/eval_structured_curriculum_ranges.sh \
  --mode incremental \
  --stage 0 \
  --total-num-envs 4 \
  --eval-rollout-epoch 1
```

### ImageBind RGB-distance baseline

ImageBind repo:

```text
/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/ImageBind
```

Install the lightweight inference dependencies into the current RLinf venv:

```bash
bash examples/embodiment/setup_imagebind_env.sh
```

Download the ImageBind huge checkpoint if it is not already present:

```bash
DOWNLOAD_CHECKPOINT=1 bash examples/embodiment/setup_imagebind_env.sh
```

The default checkpoint path is:

```text
/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/ImageBind/.checkpoints/imagebind_huge.pth
```

Compute RGB embeddings and single-reference cosine distance to scenario `009876`:

```bash
.venv/bin/python examples/embodiment/generate_imagebind_scenario_distance.py \
  --scenario-file rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl \
  --image-dir "logs/20260629-17:54:39-全1wsetting初始图像/scenario_initial_renders" \
  --checkpoint /inspire/hdd/global_user/gongjingjing-25039/xpyu/project/ImageBind/.checkpoints/imagebind_huge.pth \
  --output-dir rlinf/assets_isaaclab/all_setting/combined_with_imagebind_distance \
  --reference-id 009876 \
  --batch-size 64
```

Generate a static 19-stage manifest from ImageBind distance:

```bash
.venv/bin/python examples/embodiment/generate_curriculum_stage_manifest.py \
  --source rlinf/assets_isaaclab/all_setting/combined_with_imagebind_distance/Isaaclab_all_scenarios_10025_with_imagebind_distance.jsonl \
  --output-dir rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_imagebind \
  --distance-key imagebind_cosine_distance \
  --counts 50,100,200,400,600,800,989,1200,1500,1997,2500,3000,4048,5000,5964,7000,8014,9000,10025 \
  --prefix Isaaclab_19stage_10025_imagebind
```

## Training Configs and Commands

Base scenario JSONL config:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_scenarios.yaml
```

Static 19-stage curriculum:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum.yaml
```

Static 10-stage / 200 grouped-promotion curriculum:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200.yaml
examples/embodiment/run_embodiment_curriculum_10stage_200.sh
```

Run:

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf
bash examples/embodiment/run_embodiment_curriculum_10stage_200.sh
```

Dynamic top-10 10-stage / 200:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_dynamic_top10.yaml
examples/embodiment/run_embodiment_curriculum_10stage_200_dynamic_top10.sh
```

Run:

```bash
bash examples/embodiment/run_embodiment_curriculum_10stage_200_dynamic_top10.sh
```

Dynamic top-10 19-stage:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10.yaml
examples/embodiment/run_embodiment_curriculum_19stage_dynamic_top10.sh
```

Run:

```bash
bash examples/embodiment/run_embodiment_curriculum_19stage_dynamic_top10.sh
```

Max-300 forced-promotion variants:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_max300.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_dynamic_top10_max300.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10_max300.yaml
```

Run:

```bash
bash examples/embodiment/run_embodiment_curriculum_10stage_200_max300.sh
bash examples/embodiment/run_embodiment_curriculum_10stage_200_dynamic_top10_max300.sh
bash examples/embodiment/run_embodiment_curriculum_19stage_dynamic_top10_max300.sh
```

## Launch Script Changes

Modified:

```text
examples/embodiment/run_embodiment.sh
examples/embodiment/eval_embodiment.sh
```

`run_embodiment.sh` now:

- Writes launch parameters at the top of `run_embodiment.log`.
- Records config name/path, log tag, python path, GPU visibility, `nvidia-smi -L`, and exact command.
- Sets Vulkan/NVIDIA env vars needed by headless IsaacLab:

```text
NVIDIA_DRIVER_CAPABILITIES=all
VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

- Returns the true Python/RLinf exit status with `exit ${PIPESTATUS[0]}` even when piping through `tee`.

`eval_embodiment.sh` now:

- Accepts extra Hydra overrides after the config/platform args.
- Uses `HYDRA_FULL_ERROR=1`.
- Logs the exact command.

## Evaluation Tools

Script:

```text
examples/embodiment/eval_curriculum_ranges.sh
```

Purpose:

- Evaluate a checkpoint or base model over curriculum ranges.
- Supports cumulative mode and incremental mode.
- Builds per-stage scenario files under the output directory.
- Writes summary CSV/JSON under:

```text
logs/curriculum_range_eval/<timestamp-config>/
```

Typical command:

```bash
bash examples/embodiment/eval_curriculum_ranges.sh
```

With overrides:

```bash
bash examples/embodiment/eval_curriculum_ranges.sh -- \
  actor.model.ckpt_path=/path/to/checkpoint
```

Checkpoint behavior:

- If not running eval or if no ckpt path is configured, ckpt loading is skipped.
- For eval, checkpoint can be configured in YAML or passed as Hydra override.

## Rendering and Visualization

Single scenario:

```text
examples/embodiment/render_stack_cube_scenario.sh
examples/embodiment/render_stack_cube_scenario.py
```

All scenarios or subsets with fast reuse:

```text
examples/embodiment/render_stack_cube_all_scenarios.sh
examples/embodiment/render_stack_cube_all_scenarios.py
```

Important behavior:

- The fast rendering path reuses one IsaacLab app/env and applies scenario reset directly between renders.
- This avoids relaunching IsaacLab for every table USD.
- If isolation is required for unstable table USD assets, the script can fall back to isolated mode, but that is slower.

Curriculum stage-0 renderer:

```text
examples/embodiment/render_stack_cube_curriculum_stage0.sh
```

Incremental stage sample renderer:

```text
examples/embodiment/render_stack_cube_incremental_samples.sh
```

Purpose:

- Render about 20 examples per incremental stage for manual inspection.
- Supports input directories such as:

```text
rlinf/assets_isaaclab/all_setting/curriculum_10stage_200/stage_sets/incremental/
rlinf/assets_isaaclab/all_setting/curriculum_10stage_200_dynamic_top10/stage_sets/incremental/
rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_dynamic_top10/stage_sets/incremental/
```

## GPU Smoke Test

Script:

```text
examples/embodiment/check_isaaclab_gpu_smoke.sh
```

Purpose:

- Reuse the training pipeline enough to test whether a GPU can:
  - launch IsaacLab,
  - load JSONL scenarios,
  - load the OpenPI model,
  - generate actions,
  - step the env,
  - run a tiny actor update.

Commands:

```bash
bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 1
bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 2
bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 8
CUDA_VISIBLE_DEVICES=3 bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 1
```

Logs go to:

```text
logs/gpu_smoke/<timestamp-config>-<ngpu>gpu/
```

Important failure interpretation:

- `ERROR_DEVICE_LOST` / `gpu.foundation.plugin` crash usually indicates GPU/driver/Vulkan/VRAM instability, not a Python assertion.
- Python assertions such as `rollout_size is not divisible by batch_size_per_rank` are config/math errors, not GPU health failures.

## H100 Throughput and Memory Benchmark

Use the scaling driver to study single-node throughput before selecting an
eight-GPU configuration. It reuses the target `nearest100` PPO config and
changes only Hydra runtime overrides. Each sweep point runs a full 450-step
rollout, one actor/critic update, and records the per-GPU peak memory in
`gpu_peak.csv`.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash examples/embodiment/benchmark_isaaclab_h100_scaling.sh
```

The default two-GPU matrix is total train envs `64, 128, 192, 256`, followed
by micro batches `4, 8, 16, 32`. The default global batch is `576`: it is
`288` per rank on two GPUs and is divisible by every default micro batch. The
driver writes a machine-readable summary to:

```text
logs/h100_scaling/<timestamp>-<config>-2gpu/benchmark_results.csv
```

Run individual phases when manual selection is preferred:

```bash
bash examples/embodiment/benchmark_isaaclab_h100_scaling.sh --phase env
bash examples/embodiment/benchmark_isaaclab_h100_scaling.sh --phase micro --selected-env 128
bash examples/embodiment/benchmark_isaaclab_h100_scaling.sh --phase representative --selected-env 128
```

The `all` phase selects the best passing point by environment steps/s, then
reruns that point and a lower-memory micro-batch-4 point for two steps. Do not
linearly multiply a two-GPU winner by four: an eight-GPU batch must remain
divisible by `world_size * micro_batch_size`, and multi-node runs add NCCL
communication costs.

### qzcli submission

`submit_isaaclab_h100_benchmark_qzcli.sh` targets the cached resources for
`高岳导师课程-具身智能机器人系统` in `分布式训练空间`:

```text
compute group: 开发区-H100-cuda12.8版本-183核
spec:          2x NVIDIA_H100_SXM_80G + 40 CPU cores + 400GB memory
instances:     1
```

The confirmed RLinf 1.3 image is:

```bash
export RLINF_13_IMAGE='docker.sii.shaipower.online/inspire-studio/rlinf-xhc:1.3'
bash examples/embodiment/submit_isaaclab_h100_benchmark_qzcli.sh
```

After inspecting the dry-run and refreshing authentication with `qzcli login`,
submit the task explicitly:

```bash
bash examples/embodiment/submit_isaaclab_h100_benchmark_qzcli.sh --submit
```

The submission uses one two-GPU instance, starts a single-node Ray head with
two GPUs, verifies `torch.cuda.device_count() == 2`, and then runs the scaling
driver. Monitor it with `qzcli logs <job_id> --follow`, `qzcli events <job_id>
--all-instances`, and `qzcli worker diag <job_id>`.

## Trajectory and Replay Utilities

New/modified files:

```text
rlinf/envs/isaaclab/trajectory_recorder.py
rlinf/envs/isaaclab/replay_h5.py
rlinf/envs/isaaclab/replay_utils.py
examples/embodiment/replay_isaaclab_trajectory.py
examples/embodiment/replay_isaaclab_trajectory_batch.sh
examples/embodiment/convert_replay_h5_to_lerobot.py
examples/embodiment/merge_lerobot_datasets.py
examples/embodiment/visualize_lerobot_parts.py
```

Purpose:

- Save lightweight success/fail trajectories without image tensors.
- Replay low-dimensional state and visual scenario information.
- Convert or inspect replay outputs.

Trajectory eval config:

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_trajectory_eval.yaml
```

## Logging Conventions

Main run log:

```text
logs/<timestamp>-<LOG_NAME_TAG>/run_embodiment.log
```

Per-worker logs:

```text
logs/<timestamp>-<LOG_NAME_TAG>/worker_logs/
```

Curriculum scenario stats:

```text
worker_logs/curriculum_scenario_stats_rank_<rank>.jsonl
```

The beginning of `run_embodiment.log` should include:

```text
===== RLinf launch parameters =====
config_name=...
config_path=...
log_name_tag=...
cuda_visible_devices=...
nvidia_smi_gpu_count=...
cmd=...
===== end launch parameters =====
```

Curriculum stage changes appear as:

```text
Initialized scenario curriculum: {...}
Advanced scenario curriculum to stage N: {...}; reason=metrics
Advanced scenario curriculum to stage N: {...}; reason=max_steps_per_stage
Updated scenario curriculum stage to N: [...]
```

## Known Operational Notes

- Training uses both table camera and wrist camera as model inputs in the OpenPI config (`num_images_in_input: 2`).
- Reset snapshots usually save table camera PNG by default; wrist PNG is optional via `reset_snapshot_cfg.save_wrist_png`.
- Training does not save full videos by default; eval configs can save videos via `env.eval.video_cfg.save_video: true`.
- `GLFW initialization failed` warnings are common in headless runs and are not necessarily fatal.
- In distributed/container jobs, `conda: command not found` can appear if the shell startup path differs from interactive Docker. The scripts mostly rely on the active Python/venv path, but avoid assuming interactive shell init runs in batch jobs.
- If a wrapper script says success but the Python command failed, confirm it exits with the true command status. `run_embodiment.sh` now uses `exit ${PIPESTATUS[0]}`.
- The repository currently contains many untracked core dump files. They are crash artifacts and are not required for reproduction.

## Minimal Reproduction Checklist

1. Ensure the target machine has the same RLinf code plus the modified IsaacLab `franka_stack_events` functions.
2. Ensure `isaac_sim/` and `.venv/` or equivalent Python environment can import IsaacLab and RLinf.
3. Ensure table USD assets and any referenced PNG textures exist at the paths used by the USD files.
4. Verify JSONL scenario files exist under `rlinf/assets_isaaclab/all_setting/`.
5. Run a single-GPU smoke test:

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf
bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 1
```

6. Run an 8-GPU smoke test if distributed training is needed:

```bash
bash examples/embodiment/check_isaaclab_gpu_smoke.sh --gpus 8
```

7. Render a small sample of scenarios to verify table USD, textures, cube poses, and camera poses.
8. Start the desired training config:

```bash
bash examples/embodiment/run_embodiment_curriculum_10stage_200.sh
```

or one of the dynamic/max300 variants listed above.

9. Monitor:

```text
run_embodiment.log
worker_logs/curriculum_scenario_stats_rank_*.jsonl
tensorboard/
```

10. Confirm curriculum is progressing by searching logs for:

```bash
rg "Advanced scenario curriculum|curriculum/stage_index|reason=" logs/<run>/run_embodiment.log
```
