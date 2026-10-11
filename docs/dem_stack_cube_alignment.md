# DEM demonstration/environment protocol audit — 2026-10-10

## Reset-time image alignment resolved (2026-10-10)

The earlier issue was reproduced in the original pi05 chunkopt training
environment, scaled to two envs. It occurs before the simulation has executed
its first physics step. Extra renders and Fabric reattachment are not fixes;
reattachment even restores construction-time visuals. One initialization
physics step **before** the first full seeded reset resolves it without adding
policy/control steps or advancing physics during subsequent resets.

Shared Stack-Cube config: `init_params.startup_physics_warmup_steps: 1` (default;
set 0 for legacy comparison). `warmup_physics_before_reset` in
`rlinf/envs/isaaclab/tasks/stack_cube.py` runs once in the factory. At the current
physics dt this is 0.01s of explicitly declared pre-episode initialization.
No rotation/gripper conversion changes accompany this fix.

A/B GPU regression: six consecutive full resets at seeds 42–46 (including the
constructor reset) plus a partial reset after two control steps. Joint and cube
positions after seeded full resets match exactly between warmup=0/1. Corrected
maximum cube-projection/color-centroid error is 3.70px versus 104.84px baseline;
these centroid errors include perspective/visible-face bias. The partial
reset leaves the other env's joints unchanged and episode counters at [0,2].

A full DEM first-episode eval passes the technical video check: 450 control
steps, 451 decoded frames, 0/1 successes, no inner reset, terminal frame present,
frame-0/1 cube centroid changes below 0.58px. Visual inspection confirms that
frame 0 no longer shows the old scene. This resolves the startup/reset defect,
**not** the unknown demonstration capture hooks or original Euler conversion.

Evidence: `logs/dem_reset_fix_20261010/result.json`; reproducible GPU check:
`toolkits/isaaclab/probe_reset_alignment.py` (one GPU; `--warmup-steps 0` is the
negative control). 16 CPU regressions pass.

## Earlier empirical follow-up: training works, reset images still misaligned

The full dataset now has a 24-candidate Euler/order comparison: 147 episodes,
13,978 active-yaw transitions. XYZ + `legacy_wxyz_as_xyzw` is best on aggregate
four-prefix orientation replay error (0.05892 rad); several orders remain almost
tied on angular-increment fits. It is an empirical baseline, not unique
acquisition reconstruction. Report: `logs/dem_empirical_20261010/rotation_candidates.json`.

Real 100-step SFT and unified-factory fixed-noise equality passed. A full
450-step first episode ran (0/1 successes, no internal reset), but checking the
video revealed that its reset image still depicts the pre-reset scene. The
initial camera/control alignment is therefore **known incorrect**, not merely
unproven. This affects the first policy input as well as frame 0 in the video.

The following did not resolve it in this checkout: native reset rerenders=2,
extra render calls, scene update at dt=0, forced camera-buffer updates, and
PhysX update_transformations (fast cache and USD). All three runtime flags were
confirmed: RTX sensors present, rendering enabled, reset rerenders=2. The scene
became current after the first actual control step. This narrows the issue but
does not prove an exact backend root cause. The ineffective production rerender
option was reverted; no uncounted physics warmup was added. Reproduction patch
and diagnostic scripts live in `logs/dem_empirical_20261010/`.

The original first-episode video gate remains open. See `docs/dem_stack_cube.md`
for checkpoint and run paths; do not treat the earlier complete-decode check as
proof that frame 0 was correct.

## Earlier implementation follow-up (17:00 UTC)

The DEM input-path bug described below is now isolated and fixed: native XYZW
poses are exported only with `include_native_eef_quat=true`, preserved through
EnvOutput, and used instead of the legacy pi05 rotation vector. Existing pi05
schema/values remain unchanged. A checkpoint must explicitly declare
`state_rotation_encoding` (`native_xyzw` or hypothesis `legacy_wxyz_as_xyzw`);
missing native observations or missing/unknown encodings are rejected. This
repairs transport and mathematical conversion, **not acquisition provenance**.

15 CPU tests pass. A new real-GPU probe completed 397 steps with identical
seed-42 trial resets, native roundtrip error below 1.20e-7 rad, and episode
0/20/70/120 prefixes. Candidate WXYZ-as-XYZW mean orientation errors were
0.0243/0.1086/0.0613/0.0415 rad, versus approximately pi for native Euler.
Different demonstration initial states and possible collection frame conventions
remain confounders. Artifacts: `logs/dem_native_pose_mytest_20261010/`.
The dataset Community discussion is only a generic Parquet-conversion bot post;
no original acquisition/export conversion has been found. No model was trained.

## Earlier audit outcome (14:42 UTC)

**Acceptance item 1 is not complete.** Gripper/control scale and runtime timing
are measured. Rotation correspondence has a confirmed legacy-wrapper problem
and an unresolved dataset-conversion convention. No change was made to the
working pi05 observation path, and no trained-policy acceptance is claimed.

Resource: user-selected `mytest`, `qb-prod-gpu038`, H100 80GB, driver 595.58.03,
image v1.8. Final controlled probe exited 0 after 333 control steps. Its processes
exited and GPU memory was 0MiB afterward; the Notebook was left running.

## Evidence and reproduction

- Probe: `toolkits/isaaclab/probe_dem_contract.py`.
- Final artifacts: `logs/dem_alignment_mytest_20261010/probe_fixed/`.
- Actual IK calls: `probe_fixed/ik_trace.jsonl` (5 physics substeps/control step).
- Offline assertions/report: `logs/dem_alignment_mytest_20261010/analyze.py`;
  run `.venv-dem/bin/python logs/dem_alignment_mytest_20261010/analyze.py probe_fixed`.
- Full demonstration statistics: `logs/dem_alignment_mytest_20261010/dataset_timing.json`.
- Probe uses all six pose axes with both signs, gripper +1/-1/0, and episode 0's
  first 64 raw actions replayed with multipliers 0.5, 1, 2 on pose only.
- Every trial resets with seed 42. All 16 measured initial 8D states were identical;
  they were **not identical to the recorded demonstration's initial state**.
- Earlier `probe/` was stopped by an overly strict roundtrip assertion based on
  the old, incorrect WXYZ interpretation. `probe_native/` corrected quaternion
  interpretation but had differing random resets. Neither is the final controlled
  comparison; both are retained as diagnostic history.

## 1. Gripper: runtime and demonstration signs agree

Runtime observations preserve `[q_left, -q_right]`, in metres. Measured fingers:

| Applied command | After 15 control steps |
| --- | --- |
| +1 | approximately `[0.04, -0.04]` (open) |
| -1 | approximately `[0.00011, -0.00011]` (closed) |
| 0 | approximately `[0.04, -0.04]` (open, **not hold**) |

Across all 147 demonstration episodes, negative switches (377 events) reduce
signed aperture by 0.03798m on average over 5 frames; positive switches (379)
increase it by 0.04071m. Object contact can prevent full closure.

DEM's action helper keeps negative=close and zero/positive=open. No second
sign inversion or conversion to a continuous aperture target is needed.

## 2. Actions: raw relative IK, scale applied once

For all 333 measured first-substep controller calls:

- `processed_arm_action == raw_action[:6] * 0.5`, maximum error 0.
- `target_position = current_TCP_position + processed[:3]`, maximum error
  `2.98e-8 m`.
- `target_rotation = Exp(processed[3:6]) * current_TCP_rotation`, maximum
  geodesic error `1.89e-8 rad`, treating native controller quaternions as XYZW.
- Rotations are **angle-axis/exponential-coordinate increments**, not general
  additive Euler angles. They left-multiply in the robot-base frame.
- Observation frame offset is 0.1034m from `panda_hand`; IK offset is 0.107m.
  These points must not be assumed identical when auditing rotation-induced motion.

Metadata saying roll/pitch/yaw does not override actual controller semantics.
Dataset actions 3 and 4 are always zero, so yaw-only commands cannot distinguish
an Euler-increment convention from angle-axis. Do not invent evidence for those
unexcited axes.

Controlled 64-action prefix replay, comparing per-step position increments at
lag zero against the demonstration:

| Extra multiplier before the environment | Position-increment RMSE |
| --- | --- |
| 0.5 | 1.566 mm |
| **1.0 (raw, no extra scaling)** | **0.316 mm** |
| 2.0 | 3.280 mm |

This supports passing stored actions unchanged. It is not an exact reconstruction
of data collection: starting states differ by approximately `[4.34,-2.46,4.40]`
mm and collection simulator/dynamics provenance is missing. A raw 0.02 command
creates a 0.01 target increment, not a promise of 0.01m actual motion in one step.

## 3. Time: 20Hz runtime and no evidence for shifting actions

Measured `physics_dt=0.01`, `decimation=5`, hence `control_dt=0.05s` (20Hz).
Every returned `observation.actions` exactly matches the action just executed.
The rollout contract is therefore `obs_t -> action_t -> obs_(t+1)`.

All dataset camera frame counts and PTS were checked previously. Across 53,265
rows, correlate `s(t+1)-s(t)` with `action(t+lag)`:

| lag | position correlation |
| --- | --- |
| -2 | 0.91744 |
| -1 | 0.97674 |
| **0** | **0.98553** |
| +1 | 0.92835 |
| +2 | 0.85131 |

The fixed-seed replay also best matches lag zero. Keep training sample
`obs[t] -> actions[t:t+H]`, without a guessed one-frame shift. Statistical alignment
and matching PTS do **not** independently prove acquisition-side pre/post-step
hook placement or camera latency; the original converter/recorder is still needed.

## 4. Rotation: unresolved, do not accept the current DEM transform

### Confirmed runtime issue

This checkout's IsaacLab 3 uses native XYZW quaternions. Evidence includes
`isaaclab_physx/sensors/frame_transformer/frame_transformer_data.py`, which
initializes identity at component 3, Warp `quatf` tensors, and the measured IK
rotation targets above.

`rlinf/envs/isaaclab/tasks/stack_cube.py::_wrap_obs` still assumes WXYZ and reorders
`[1,2,3,0]` before its rotation-vector conversion. It is a legacy pi05 model input
convention, **not the current simulator's physical orientation**. The current
pre-fix provisional DEM `environment_state_to_dataset` converted that legacy vector to
Euler without undoing this discrepancy. Runtime orientation error versus the
native pose approaches pi radians. Its CPU tests only check mathematically valid
rotation-vector inputs; passing those tests did not validate this upstream assumption.

The pi05 path was deliberately preserved: its existing checkpoint may have learned
these features, and changing them globally could invalidate its working pipeline.
That unsafe input dependency is now removed as described above; dataset-encoding
acceptance is still pending.

### Dataset uncertainty

`meta/modality.json` names roll/pitch/yaw, but provides no quaternion order or
conversion formula. Stored orientations disagree with native XYZ Euler by about
pi radians during prefix replay. Supplying native XYZW as `[w,x,y,z]` to SciPy's
XYZW parser *before* Euler conversion is a substantially closer hypothesis:
mean error approximately 0.0243 rad at raw-action replay. Other permutations can
also be close over this restricted trajectory. This is not proof of a particular
conversion bug or a license to silently change all dataset rotations.

Across the full dataset, yaw action correlates more with the X component of
rotation increments under direct XYZ Euler interpretation (0.654) than the Z
component (0.271). Undoing the suspected WXYZ-as-XYZW ordering moves the strongest
correlation to Z (0.653). This supports investigating the conversion order, not
marking its exact convention as verified.

Searched local sources, dataset metadata, official RLinf task docs, and the
public `YifWRobotics/RLinf-pi0.5-IsaacLab` tree; the original HDF5-to-LeRobot
conversion for this dataset was not located.

## Next required evidence

Obtain the original collection/export script, specifically
`eef_quat -> observation.state[3:6]`, or a raw HDF5 episode with its matching
Parquet. Confirm quaternion order, Euler convention, frame/TCP offsets and
pre/post-step recording order. Then implement a DEM-only native-pose adapter
and matching dataset conversion with a regression test using paired real records.
Until then keep `contract_verified=false`; do not start formal acceptance
training with a guessed rotation correction. Existing 14 CPU regression tests
pass, and the expanded GPU probe's runtime assertions pass.
