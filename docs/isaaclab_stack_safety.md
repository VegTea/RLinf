# Nearest100 instanceable: safety-termination training variant

Config: `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100_instanceable_safety.yaml`.
It copies the configuration used for
`logs/20260901-19:59:50-h100-8x-stack-cube-nearest100-instanceable-isaacsim6`:
128 training environments, 450 steps per rollout, 5 actions per chunk, nearest100
instanceable scenarios and the same model/PPO settings. The original config is
unchanged. This is a new run, not a checkpoint resume.

## Added training failure conditions

Both return `terminated=True`, not a time-limit truncation:

- **Joint-limit stall:** the same one of the seven arm joints is within 0.02 rad
  of a hard position limit, its commanded joint-position target pushes farther
  outward by more than 0.005 rad, and its speed is at most 0.05 rad/s, for five
  consecutive control steps (0.25 s). Being near a limit alone is not a failure.
  Gripper joints are excluded. The limits come from the actual robot asset.
- **Self-collision:** the same monitored robot body pair has normal contact force
  greater than 5 N for two consecutive control steps (0.10 s). One sensing body
  per contact sensor is used, with explicit robot-only partner filters. Adjacent
  arm links, the fixed link7/hand pair and gripper-internal contacts are excluded.
  Cube/table contacts do not count. Contact reporting and articulation
  self-collisions are explicitly enabled. Existing USD collision filtering is
  retained; pairs disabled in the asset cannot produce measured contacts.

Counters are independent per environment and per joint/contact pair; they clear
on a broken condition or an episode reset, including resets from other causes.
Contact checks use force samples at control-step boundaries, not continuous
collision detection. Very brief contacts between samples are not covered.
Thresholds are initial calibration values, not a validated real-robot safety
system. Tune the values under `env.train.init_params.safety_terminations`.

Success, cube-drop and timeout conditions remain enabled. Reward is unchanged
(no extra failure penalty). Evaluation deliberately retains the original config;
to compare safety during evaluation, a separate aligned evaluation configuration
is needed. The baseline training config disables periodic evaluation.

## Best-only checkpoints

This safety config now sets `runner.save_best_checkpoint: true` and
`runner.save_interval: -1`. After each rollout, a strictly higher training
`env/success_rate` (first-episode success) saves the **measured policy before the next PPO update**.
The first valid score saves `global_step_0` (the initial SFT policy). This is not
an evaluation-set best or a minimum-loss checkpoint. Equal/lower scores do not
save. Only the latest best produced by this run is retained; publication completes
before the prior best is removed. Periodic and final-step saves are disabled.

Location: `<log_path>/<experiment_name>/checkpoints/best/`. `best.json` records
score, step and checkpoint path. Each best contains the full actor weights,
DCP training state and `runner_state.json`. W&B only uploads logs, not checkpoints.

## Reset, rollout and PPO masking

`auto_reset: false` / `ignore_terminations: false` preserve the baseline RLinf
training behavior. Isaac Lab still resets individual environments immediately
inside its step after a termination. RLinf continues the current action chunk and
the 450-step rollout, and resets all environments at the next rollout start.
Safety training configs enable `env.train.init_params.scenario_reset.freeze_within_rollout`:
internal resets reuse each slot's assigned scenario; explicit outer resets at the
next rollout epoch sample again. Partial outer resets only reassign the selected
slots. `freeze_per_env: true` still overrides this for permanent evaluation replay.
The subprocess marks outer resets on the unwrapped simulator in a `try/finally`,
so resetting does not leave the marker enabled after an exception. This preserves
scenario-defined cube/table/camera settings, not a bitwise simulator snapshot or
unrelated visual randomization. Configurations without the new flag are unchanged.
The actor's existing loss mask excludes post-first-done steps; with chunk-level
losses the terminal chunk is still retained. Before collapsing this mask to chunk
level, the actor now zeros rewards after the first done. The terminal action's
reward is retained. For example, rewards `[0, 0, 1, 0, 0]` with failure on action 2
now produce chunk reward 0, not 1 from the reset episode. This also happens before
optional reward filtering. The fix applies to the shared embodied FSDP actor when
`auto_reset: false`, `ignore_terminations: false`, and `reward_type: chunk_level`;
other modes are unchanged.

GAE still stops at a terminal chunk; fully masked chunks remain excluded from
losses and advantage normalization. Each rollout epoch gets its own first-done
mask. Current timeout handling is unchanged: truncations also end the return,
without terminal-value bootstrap in this mode. This corresponds to treating the
450-step task budget as an episode endpoint, not a continuing-task sampling cut.
TODO(agent): if timeouts instead represent sampling cuts, preserve terminal
observations and implement timeout bootstrap before using that interpretation.

This fix does **not** stop remaining chunk actions from executing in an internally
reset scene, or redesign terminal observations. Post-first-done sampling still
wastes computation; more frequent early failures can reduce useful samples.
Already-running workers must restart to load the fix. Existing checkpoints remain
loadable, but previous contaminated updates cannot be undone by resuming them.

CPU regression checks (no model, simulator or GPU launch):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv-model312/bin/python -m pytest tests/unit_tests/test_embodied_first_episode_rewards.py -q
```

## First-episode success metric

`env/success_rate` is the fraction of environment
slots whose **first episode after the outer reset** succeeds. It uses the same
positive raw task reward signal as `success_once`, but freezes each slot at its
first termination or truncation, including the outer time limit. Success on the
terminal transition counts. Subsequent Isaac Lab internal resets cannot reopen
that slot's success counter, even within a five-action chunk. Explicit outer
resets clear the counters only for the reset slots.

For the current `auto_reset: false` training setup, the worker retains the final
metrics for each rollout epoch and the runner averages the per-slot Boolean
values across ranks/epochs. Thus two rollout epochs with 128 environments give
256 first-episode outcomes per training iteration. Until a slot ends, it records
whether success has occurred so far; this is not a completed-episodes-only ratio.

The original `env/success_once` still measures any success within the outer
rollout window and can include later episodes after internal resets. The metric
itself does not change rewards, terminations or loss masks; PPO reward masking is
described separately above. Best-checkpoint selection now uses
`env/success_rate`; a higher `success_once` alone cannot replace the best.
When resuming legacy checkpoints, model/optimizer/step recovery is unchanged,
but a stored best score for `env/success_once` is ignored rather than compared
against the new metric. Legacy best weights are retained until a new-metric best
is safely published; external source-run checkpoints are never deleted.
Running Python workers do not hot-reload these changes: they become available
on the next launch/restart that imports the updated code.

CPU regression check:

```bash
PYTHONPATH=. .venv-model312/bin/python -m pytest tests/unit_tests/test_isaaclab_first_episode_success.py -q
```

## Launch

From the repository root, on the intended 8-GPU node, with a Ray head exposing
those eight GPUs (do not reuse a differently sized Ray cluster):

```bash
source starts/source.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOG_NAME_TAG=h100-8x-stack-cube-nearest100-instanceable-safety-isaacsim6
bash examples/embodiment/run_embodiment.sh \
  isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100_instanceable_safety \
  LIBERO
```

Example threshold override: append
`env.train.init_params.safety_terminations.self_collision.force_threshold=10.0`.

## Offline GPU logging with CPU live upload

Enable `runner.logger.logger_backends=[tensorboard,wandb]` and set
`WANDB_MODE=offline` **before starting Ray** on the GPU node. The CPU container
must see the same shared log directory and have its own W&B login. No W&B
credentials or network access are needed on the GPU node.

Once the main `.wandb` file appears, run on the CPU container (W&B 0.28.2):

```bash
tmux new-session -d -s wandb-sync \
  '.venv-model312/bin/wandb beta sync --live --yes --entity vegtea-hust --project rlinf /absolute/path/to/run-ID.wandb > /absolute/path/to/sync.log 2>&1'
tmux list-panes -t wandb-sync -F '#{pane_pid}'
tmux attach -t wandb-sync
```

With per-worker logging, the main file is under
`<log_path>/wandb/all/wandb/offline-run-*/run-*.wandb`. Only this aggregate run
needs live upload; worker runs remain on disk. `MetricLogger` keeps independent
W&B Run handles (`reinit=create_new`) so creating worker runs cannot finish the
main run or redirect its metrics. TensorBoard paths are unchanged.

The existing runner logs several metric dictionaries with the same explicit W&B
step. W&B commits that row when the next step is logged (or the run finishes),
so the live cloud history can lag by one training iteration even when the CPU
sync is healthy. TensorBoard already contains the just-completed iteration.

Tmux survives terminal disconnection, not CPU-container shutdown. If either the
container or network fails, local GPU logs remain available for later sync. The
native `--live` command may wait indefinitely after an unclean training crash;
check the platform task state before deciding to stop/restart the CPU sync.

## Validation

CPU-only logic and wiring check (no Kit/GPU required):

```bash
PYTHONPATH=. .venv-model312/bin/python tests/unit_tests/test_isaaclab_safety_terminations.py
```

Before a long training run, perform a short Isaac Sim smoke run and confirm the
termination manager lists `joint_limit_stall` and `self_collision`. Inspect sensor
forces in a deliberately colliding pose and a normal cube grasp, and verify that
only the former triggers. CPU checks do not validate PhysX contact reporting,
asset collision coverage, performance, or the chosen force thresholds.
