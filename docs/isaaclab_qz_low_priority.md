# QZ four-H100 low-priority training

Config: `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100_instanceable_safety_lowprio_4gpu.yaml`.
It retains nearest100 instanceable scenarios, both safety termination terms,
SFT initialization and seed 42. It uses 64 training envs / 16 eval envs,
32 micro-batch / 128 global batch, preserving the eight-card run's per-GPU load.
W&B is offline on GPU; TensorBoard and worker logs remain enabled.

## Checkpoint policy

```yaml
runner:
  save_best_checkpoint: true
  save_interval: 1
  keep_last_n_checkpoints: 2
  checkpoint_dir: null  # launcher supplies <RUN_ROOT>/checkpoints
  resume_dir: auto
```

- Save one periodic checkpoint **after each PPO iteration**. Adjust `save_interval`
  if this write frequency is too expensive. Only the two newest complete periodic
  checkpoints in this experiment remain; the third successful save deletes the oldest.
- Keep the best training `env/success_rate` (first-episode success) policy separately
  in `checkpoints/best/`. It is measured/saved before the next PPO update and is not
  an evaluation-set score. Legacy `env/success_once` best scores are not comparable:
  resume training state as usual, but establish a fresh best under the new metric.
  Retain legacy best weights until the new best is successfully published.
- Write into `.saving-*`, wait for the distributed actor save and write `.complete`,
  then rename into `global_step_N` on the same filesystem. Only after publication
  may retention delete old checkpoints. A preempted/incomplete staging directory
  is never selected for resume or counted against retention.
- Retention works across process restarts. It only scans **marked periodic siblings
  of this experiment**: it does not recurse into `best/`, follow checkpoint symlinks,
  delete unmarked legacy directories, or touch checkpoints in another experiment.
  In-flight/stale `.saving-*` directories are left for manual inspection after an
  unclean kill, rather than being mistaken for valid checkpoints.
- Automatic resume selects the newest complete periodic checkpoint, otherwise a
  complete best, otherwise starts from the SFT model. Models, optimizer states,
  LR scheduler and global training step are restored. Best score/ownership is
  restored too, so subsequent improvement replaces the same experiment's old best.
  Legacy checkpoints without `.complete` require an explicit `runner.resume_dir`.
- This is **training-state recovery**, not bitwise replay: simulation state and RNG
  streams are not restored. Unsaved work since the last complete checkpoint is lost.

## Launch and restart

Choose one absolute `RUN_ROOT` and **reuse it on every retry of this experiment**.
The launcher creates a new `attempts/<timestamp>.<suffix>/` for logs each time,
while keeping the checkpoint directory stable. An exclusive shared-file lock
prevents two attempts from writing/pruning that tree concurrently.

```bash
# In a newly allocated four-GPU container:
bash starts/train_h100_4x_nearest100_safety_lowprio_isaacsim6.sh \
  /inspire/.../logs/my-lowprio-experiment
```

Submit through QZ using the user-selected image
`docker.sii.shaipower.online/inspire-studio/vegtea-dev-cc-conda-tailscalevac:v1.7`,
workspace `分布式训练空间`, project `公共科研项目`, full compute-group name
`训练区-H100-cuda13.2版本-183核`, quota `4,80,900`, and **`--priority 1`**.
Use `--shm-size 900`. Re-query live quota and image state before submission.
The job command must use the absolute launcher and RUN_ROOT paths.

**Automatic resume does not mean automatic reallocation.** Do not assume platform
fault-tolerance restarts a preempted terminal Job. After preemption, submit a new
Job name with the **same launch command / RUN_ROOT**. The launcher resumes without
having to pick a checkpoint manually. Confirm the old Job is terminal first.
This workflow does not stop or reuse the separate eight-GPU experiment.

## CPU-side W&B live sync across retries

Start once on the network-enabled CPU container, using the same RUN_ROOT:

```bash
tmux new-session -d -s wandb-lowprio \
  '.venv-model312/bin/python -u toolkits/isaaclab/sync_lowprio_wandb.py /inspire/.../logs/my-lowprio-experiment --merge-attempts > /inspire/.../logs/my-lowprio-experiment/wandb_supervisor.log 2>&1'
```

With `--merge-attempts`, all main streams upload to **one cloud run**, using native
`wandb beta sync --id`. The supervisor persists the earliest main run's ID and
entity/project in `<RUN_ROOT>/wandb_merged_run.json`; changing entity/project is
rejected rather than silently splitting the experiment. Training and checkpoint
logic are unchanged, and each attempt still has separate local files.

The supervisor backfills older attempts chronologically (including files already
uploaded to other run IDs), then live-syncs the current one. When `active_attempt`
changes, it stops the old live client and exports that attempt's persisted main
TensorBoard scalars into a **clean, finished offline W&B fragment**. It uploads
this fragment without `--live`, then follows the new attempt. Preempted raw
`.wandb` files can have truncated tails (`unexpected EOF`) and may also lack the
last buffered history row; TensorBoard recovery avoids those problems. Both
original formats remain untouched. Historical charts should use the global
Step axis: recovered fragments have new upload timestamps. No missing metrics
are invented when neither format recorded a step.

Target-specific completion markers prevent needless backfill on supervisor
restart; a partially uploaded current fragment can replay. A normal `exit_status`
or an operator-written `wandb_source_closed` file (only after confirming the GPU
attempt has terminated) also finalizes the current attempt. Only main scalars
are recovered; per-worker streams are never merged into the main run. W&B CLI
0.28.2 may report `wandb: ERROR` with exit code zero, so the supervisor checks
both the process exit code and its new log output before marking success.

The runner supplies global training steps. Native W&B 0.28.2 sync was checked with
overlapping fragments and a repeated upload: steps remained 0, 1, 2, with the
later fragment's overlapping value replacing the earlier value. There is no
artificial step offset. This is a single per-step curve, not a rollback audit
trail; local fragments retain the original records. Historical per-attempt cloud
runs are **not deleted**. Without `--merge-attempts`, legacy separate-run behavior
remains available. Do not run two sync clients against the unified target at once.
The stable supervisor PID is in `wandb_sync_supervisor.pid`; the current upload PID
is in `<active_attempt>/wandb_sync.pid`. Credentials are only required on CPU.
Tmux survives terminal disconnection, not CPU-container shutdown. As in the original
run, W&B's same-step buffering can delay cloud history by one training iteration.

## Checks

```bash
PYTHONPATH=. .venv-model312/bin/python -m pytest \
  tests/unit_tests/test_embodied_runner_best_checkpoint.py \
  tests/unit_tests/test_qz_lowprio_launcher.py -q
```

The checks cover save/publish failures, retention across restart, best isolation,
legacy/partial/symlink exclusion, automatic latest/best/empty selection, and the
CPU sync client's handoff/cleanup. An actual four-GPU smoke run is still needed to
validate scheduling, GPU capacity, and checkpoint I/O on the selected node.
