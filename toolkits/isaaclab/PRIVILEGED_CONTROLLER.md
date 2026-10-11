# Single-scenario privileged stacking (simulation only)

No pi05, checkpoint, learned policy, or Ray. Reads actual cube poses and robot
joints. Reuses the Stack Cube environment/scenario factory, plans Cartesian
waypoints with USD-derived bounded numerical IK, then sends joint-position
references to the existing actuators; the gripper keeps its original action term.
Near-base grasps use a configurable tilted approach (`--near-tilt`, default 35°).
The goal remains blue → red → green, checked using `cubes_stacked` after release.
Placement uses the measured cube-to-TCP rotation to keep the cube (not necessarily
the gripper) upright, preserving its original yaw. Success-triggered auto-reset
is disabled only to allow retreat and a 60-step stability check; the original
success predicate is still evaluated every step. Drop/time-limit terminations remain.

```bash
source starts/source.sh
CUDA_VISIBLE_DEVICES=0 .venv-isaacsim6/bin/python \
  toolkits/isaaclab/privileged_stack_controller.py \
  --scenario /absolute/path/one_scenario.jsonl \
  --output /absolute/path/new-output-directory
```

Outputs: `video.mp4` (table/wrist side-by-side), `trace.jsonl`,
`initial_state.json`, `result.json`. Output directory must not exist.
Add `--self-test` to check zero-joint FK, an IK round trip and joint bounds without
launching the simulator. Run Ruff on the script for static validation.

This is a bounded waypoint baseline, NOT a collision-aware motion planner and
NOT suitable for hardware. Joint references stay 0.06 rad inside the USD limits;
a failed IK segment or excessive endpoint tracking error stops the attempt.
Recorded lift checks catch an empty grasp. The 1700-control-step execution budget
is longer than the learned-policy evaluation's 450 steps, so this is a feasibility
test, not an equal-budget performance comparison. `result.success` is the final
environment predicate; `success_once` is also recorded. Always inspect video for
physical validity (no interpenetration, wrong ordering, or transient stacking).

## Nearest100 vertical-path screening

`audit_vertical_grasps.py` runs on CPU using the same USD IK implementation.
Pass `--scenarios <jsonl> --usd <panda_instanceable.usd> --output <new-dir>`;
run with `PYTHONPATH=. .venv-isaacsim6/bin/python`. `--self-test` checks a
reachable descent, its joint margins, and rejection of an unreachable target.
It deduplicates positions/yaws and emits `per_scenario_cube.csv`, `summary.json`,
and full joint-path witnesses in `progress.jsonl`. Default descent uses yaw=0
and the nominal joint seed; failures receive eight extra random seeds and
canonical/cube-aligned horizontal yaws. Orientation remains vertical throughout.

This is a sampled, joint-limited **IK screen**, not proof of physical graspability
or impossibility: no self/table/object collisions, gripper contact, dynamic
limits, or initial approach are checked. A path found under a different IK
branch need not be reachable from the actual robot configuration. No path found
means only this bounded search failed. Blue cubes are included in the report,
although the stacking task only needs to pick red and green. Scenario IDs are
not transient vector-environment indices.
