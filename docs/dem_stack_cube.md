# DEM Stack-Cube SFT and first-episode evaluation

## Reset first-frame fix verified (2026-10-10)

The shared Stack-Cube factory now performs **one initialization physics step
before the first full seeded reset**. The common config explicitly sets
`init_params.startup_physics_warmup_steps: 1` (0 reproduces legacy startup).
This applies to both pi05 and DEM environments using that config. It does not
call `env.step`, feed a policy action, or increment an episode counter; the
subsequent seeded reset restores positions/velocities. It is not an extra step
on every reset. The existing pi05 rotation/gripper conversion is unchanged.

Controlled A/B using the original pi05 IsaacSim6 chunkopt **training environment**
scaled to two envs confirms the issue also exists there. Across six consecutive
full resets (no control steps between them) and one partial reset, the maximum
projected-cube-center vs colored-pixel-centroid error was **104.84 px with startup
warmup off, 3.70 px with it on**. This is a coarse image/geometry alignment check,
not subpixel calibration. Seeded joint/cube positions were exactly equal across
A/B runs; selected episode counters reset to zero and the unreset env retained
its joints and episode length. Physics counters show startup once, no physics
advance during reset. This is an environment regression, not a new RL run.

DEM evaluation then completed 450 policy control steps, exit 0, success **0/1**.
Its 451-frame dual-camera video fully decodes; initial cube centroid changes
between frame 0/1 are below 0.58 px, and the contact sheet shows the correct
reset scene from frame 0. No internal reset; terminal frame available. The
**technical first-episode/video gate now passes**. This does not establish the
original dataset rotation/capture convention or demonstrate learned task success.

Evidence: `logs/dem_reset_fix_20261010/result.json`, `baseline/`, `regression/`,
`eval/`, and `video_contact_sheet.jpg`. GPU regression entrypoint:
`toolkits/isaaclab/probe_reset_alignment.py --output <new-directory>` (approved
single GPU, Isaac Sim interpreter). CPU regressions: **16 passed**.

## Earlier real experimental training and evaluation (2026-10-10)

Following approval to proceed with an **empirical compatibility convention**, we
compared 24 Euler/order hypotheses over all 147 episodes (13,978 active-yaw
transitions) and the four simulator prefix replays. XYZ plus
`legacy_wxyz_as_xyzw` has the lowest aggregate replay orientation error among
these candidates (0.05892 rad). Several Euler orders have nearly identical
angular-increment scores; the original conversion remains non-identifiable.
This is a debug baseline, not recovered acquisition provenance.

On mytest, real pretrained DINOv3 ConvNeXt-Base + NeoBERT SFT completed 100 steps,
episodes 0–3, batch 4, seed 42. All losses/gradients were finite; first/last-ten
mean losses were 1.3973 / 0.6878 (not a task-success claim). Deployment tensors
exactly match the final training state, the initially-zero head output weights
changed, and the **unified RLinf model factory** produced identical actions to
native DEM under fixed noise (maximum absolute difference 0.0).

Checkpoint: `/inspire/hdd/global_user/czxs24230043/pretrained_models/DEM/StackCube/smoke-empirical-20261010/policy_step_100.pt`.
`contract_verified` remains false; both training and evaluation used explicit
debug overrides. Actual resume continuation has not been tested.

RLinf ran a complete 450-control-step first episode, H=16/K=5/NFE=1, one env,
seed 42, exit 0, **0/1 successes**, timeout at step 450, no internal reset.
The two-camera video fully decodes (451 frames, 20 Hz, 512x256), but visual
inspection found **frame 0 depicts the pre-reset scene**. This also affects the
first policy observation. Therefore the **video-quality gate is NOT passed**.
A second eval with native `num_rerenders_on_reset=2` still failed the initial
frame check. Camera force-refresh, extra rendering, and PhysX
`update_transformations` also failed to synchronize the reset pose. The
ineffective production knob was reverted; its experiment patch and diagnostic
scripts are retained. No hidden warmup physics/action step was introduced.

Evidence: `logs/dem_empirical_20261010/result.json`, `rotation_candidates.json`,
`check_checkpoint.py`, `eval/`, `eval_fresh_reset/`, `first_frames.jpg`,
`reset_diagnostic/`, and `reset_sync_diagnostic/`. 15 CPU regressions pass.
Next: investigate reset-time GPU physics-to-render synchronization; any physical
warmup would need an explicit protocol change, not an invisible workaround.

## Earlier DEM native-pose fix (2026-10-10, 17:00 UTC)

The known input-path error is fixed **for DEM only**. Its eval config opts into
`include_native_eef_quat`; the environment exports `eef_quat_xyzw`, EnvOutput
preserves it through serialization/merging, and DEM constructs Euler features
from that native quaternion instead of pi05's legacy rotation-vector columns.
With the opt-in absent, the old observation schema and values are unchanged.

A deployment manifest must now contain `state_rotation_encoding`, either
`native_xyzw` or the explicit hypothesis `legacy_wxyz_as_xyzw`. There is no guessed
default and no fallback to legacy states. SFT requires the matching CLI option,
records it in deployment/resume files, and refuses a resume with a different or
missing encoding. Dataset state rows are not rewritten by this option.
Unverified-contract refusal remains enabled by default.

15 CPU regressions pass, including environment -> EnvOutput -> dummy DEM policy,
legacy pi05 input preservation, and missing/invalid encoding rejection. On mytest,
the real native-pose probe passed 397 control steps and four demo-prefix replays
(episodes 0,20,70,120), exit 0. Maximum native rotation roundtrip error was
`1.20e-7 rad`. The WXYZ-as-XYZW hypothesis is much closer to stored orientations
on all four prefixes (mean 0.024–0.109 rad versus 3.11–3.14 rad for native Euler),
but initial physical states do not match recorded demos, and this is **not**
acquisition provenance. Item 1 is still not fully verified; no training was run.
Evidence: `logs/dem_native_pose_mytest_20261010/summary.json` and `probe/`.

## Earlier protocol audit (2026-10-10, 14:42 UTC; input-path fix above supersedes it)

See [the measured alignment audit](dem_stack_cube_alignment.md). The final
single-GPU probe on mytest passed 333 control steps with all 16 trials starting
from identical seed-42 states. Raw relative IK scale=0.5, angle-axis increments,
gripper signs and 20Hz runtime timing are confirmed; raw-action/no-lag prefix
replay best matches the demonstration among tested variants. **Item 1 remains
incomplete:** the legacy wrapper assumes WXYZ although this IsaacLab 3 checkout
uses XYZW, invalidating the provisional DEM rotation-vector-to-Euler input path.
Dataset conversion order is also unresolved. No pi05 observation behavior was
changed; keep the deployment contract unverified pending original collector/
converter code or paired raw records. Both towers remain downloaded and tested.

## Successful pi05 control run on mytest (2026-10-10, 12:20 UTC)

The existing pi05 eval completed on user-selected `mytest`, node `qb-prod-gpu038`,
with one H100 80GB and driver 595.58.03 (CUDA 13.2). Step 319, seed 42, four
environments, 450 steps each, DLAA, and dual-camera rendering were retained.
Exit code was 0; legacy fixed-window `success_once` was 3/4. All four MP4s decoded
successfully (452 frames each, 20fps, 512x256); sampled images show both cameras
and changing scene content. Evidence is in `logs/pi05_eval_mytest_20261010/`,
including `result.json`, `video_validation.json`, and `video_contact_sheet.jpg`.
This is a pi05 runtime check, not DEM training or first-episode-only acceptance.

The preceding public-node eval used the same launch arguments apart from its
output directory and failed before reset. However, `mytest` also changes the
container image from v1.7 to v1.8 and shared memory from 32 to 200GiB. Do not call
physical node failure the isolated root cause without controlling these differences.
H100/DLAA alone cannot explain the failure. Test processes and Ray were cleaned up;
`mytest` itself was left running as user-owned infrastructure.

## Backbone and earlier probe verification (2026-10-10, 10:09 UTC)

- Both pretrained towers are now available under `pretrained_models/DEM/`.
  DINOv3 ConvNeXt-Base was downloaded from ModelScope (350,302,312 bytes,
  SHA256 `ec90bd798b5fc5b8e30443796a6c24a7a73e28ad85c6c0ceda78b1d249a694cc`).
  NeoBERT weights were completed from ModelScope (980,567,608 bytes,
  SHA256 `b547ae956c4ef7024a444d93464a9028cc6594cf36830fd5069a7ef9a53d799f`);
  the existing config matches that mirror's SHA256. Repository file revisions and
  hashes are saved beside the weights. These are ModelScope sources, not a claim
  that the original Hugging Face access request was approved.
- CPU pretrained forwards passed: DINOv3 `[1,3,256,256] -> [1,256,512]`;
  NeoBERT strict loading and text forward `[1,7,768]`, all finite. Per-tower
  `dem-load-check.json` records the checks. No policy training has been run.
- New public-project H100 node `qb-prod-gpu163`, driver 595.58.03 / CUDA 13.2,
  reproduced `VkResult: ERROR_DEVICE_LOST` during RTX initialization, before reset.
  No camera image or control-step result was produced. Probe processes were
  terminated and GPU memory returned to 0MiB; the Notebook itself was left running
  with its pre-existing four-hour auto-stop timer.
- Evidence: `logs/dem_contract_probe_public_20261010/{run.log,result.json}`.
  NVIDIA's [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
  explicitly exclude GPUs without RT Cores, including H100/A100, but this is NOT
  an established explanation of this crash and does not mean H100 cannot run this
  project's workload. Historical counterexample: the successful 2026-09-24 run
  under `logs/20260923-084432-h100-8x-lowprio-safety/attempts/20260924-063320.jjMrKH/`
  records actual H100 80GB, Vulkan, the same driver 595.58.03, and completed training
  metrics (`success_once=0.7734375`). Its image is also v1.7. Training used
  antialiasing Off and chunk-end rendering, unlike this DLAA probe. These are
  comparison candidates, not proven causes: the crash is early in AppLauncher,
  before task render overrides are applied. Reproduce the known-working startup
  and inspect runtime differences before concluding a GPU change is necessary.

## Earlier implementation checkpoint (2026-10-10; superseded where noted above)

This is **experimental integration, not a verified trained policy**. No real DEM
training checkpoint or policy-evaluation video has been produced.

- Full public dataset downloaded: revision
  `2e27f04b2e31f293fd020164917e4cce09e108d6`, 147 episodes, 53,265 rows,
  294 AV1 videos. All decoded frame counts and video/parquet timestamps checked.
- H=16 yields 51,060 unpadded samples; the last 15 starting positions of each
  episode are deliberately excluded. Statistics count original frames once.
- State is strongly consistent with Euler XYZ: maximum consecutive orientation
  change across all episodes is 0.03995 rad under that interpretation versus
  3.00591 rad under a rotation-vector interpretation. This is **not acquisition
  provenance**. Exact rotation order, action units and observation/action alignment
  still need collection-code or simulator confirmation.
- Finger observations preserve `[q_left, -q_right]`. Recorded gripper commands are
  {-1,+1}; all arm commands lie in [-0.1,0.1], and action dimensions 3,4 are zero
  throughout this dataset. The installed controller uses relative pose, scale=0.5,
  20 Hz, negative gripper=close and zero/positive=open. Never multiply by 0.5 twice.
- DINOv3 official download is gated (403); user confirms approval is pending.
  **Do not substitute random towers and call that a pretrained DEM base model.**
- An authorized one-H100/80GB, 20 CPU, 200GiB instance was created. The actual
  Isaac Sim 6 probe on driver 595.58.03 crashed in renderer initialization with
  `VK_ERROR_DEVICE_LOST`, before task reset. No semantic/simulation validation can
  be claimed from that run. The instance was stopped after verifying no tasks
  remained; its shared artifacts remain available.
- The 14 project CPU tests listed below pass. Dummy-policy unit tests are not
  checkpoint-training evidence. The full upstream MeanFlow CPU test invocation
  timed out and is not counted as passed.

Evidence:

- `logs/dem_data_audit.json`: full episode audit.
- `logs/dem_contract_probe/run.log`: renderer crash and allocated device details.
- Feishu plan: <https://sii-czxy.feishu.cn/docx/SLVxdBNANorFTFxn6fdcBc4DnEe>.

## Environment and weights

Do not upgrade `.venv-model312` or `.venv-isaacsim6`. Driver remains in the existing
model environment; DEM rollout runs in `.venv-dem`, sim in `.venv-isaacsim6`.

```bash
# CPU preparation; no GPU process is launched by installation.
git clone https://github.com/Apollo-Lab-Yale/decoupled-embodiment-model third_party/DEM
git -C third_party/DEM checkout e5c8f6a81dca59bdfea2d1de067a7ec6f8a3e0ea
uv venv --python .venv-model312/bin/python .venv-dem
uv --no-config pip install --python .venv-dem/bin/python -r requirements/embodied/dem.txt
uv --no-config pip install --python .venv-dem/bin/python --no-deps -e third_party/DEM
```

`--no-config` avoids the repository's legacy OpenPI UV overrides silently
replacing explicitly requested package versions. Ray is pinned to 2.52.1, matching
existing workers. The upstream model repository is separately cloned, not vendored;
check its usage/license terms before redistribution.

Default weight root: `/inspire/hdd/global_user/czxs24230043/pretrained_models/`.
Suggested classification:

```text
pretrained_models/DEM/
  DINOv3-ConvNeXt-Base/  # authorized upstream config + model.safetensors
  NeoBERT/              # config + tokenizer + model.safetensors
  StackCube/<run>/      # policy_step_N.pt + .json, training.pt, logs
```

Pin DINOv3 revision `e619358a5fe272554f17d2fb6cc65a679326bac0` and NeoBERT
`5424c8efeea6491b151d62dee55a752165407430`. Neither a directory existing nor
config/tokenizer files alone prove weights are ready. NeoBERT metadata/tokenizer
were downloaded, but its large weight download has not completed.
Public data is at `/inspire/hdd/global_user/czxs24230043/data/IsaacLab-Stack-Cube-Data`.

## Offline data audit

```bash
.venv-dem/bin/python -m rlinf.data.datasets.dem_stack_cube \
  --data /inspire/hdd/global_user/czxs24230043/data/IsaacLab-Stack-Cube-Data \
  --output logs/dem_data_audit.json
```

The adapter uses PyArrow/PyAV directly, without OpenPI padding, normalization or
a second delta transform. Camera order is front/table then wrist. It decodes RGB
without vertical flipping and verifies each video timestamp against the parquet
row. Raw state/action units reach DEM's normalizers. Full-dataset random access
may be decode-bound: an 8-video LRU is the deliberate small-run memory ceiling.

## Training entrypoint (not yet GPU-validated)

Check/approve GPU resources before executing this command. Do not run while
DINOv3 access or the control contract is unresolved for a formal experiment.

```bash
export PYTHONPATH="$PWD"
WEIGHTS=/inspire/hdd/global_user/czxs24230043/pretrained_models/DEM
.venv-dem/bin/python examples/sft/train_dem.py \
  --data /inspire/hdd/global_user/czxs24230043/data/IsaacLab-Stack-Cube-Data \
  --vision-model "$WEIGHTS/DINOv3-ConvNeXt-Base" \
  --language-model "$WEIGHTS/NeoBERT" \
  --episodes 0,1,2,3 --steps 100 --batch-size 4 --seed 42 \
  --out "$WEIGHTS/StackCube/smoke-seed42" \
  --state-rotation-encoding legacy_wxyz_as_xyzw \
  --allow-unverified-contract
```

The selected rotation encoding is a **hypothesis-only debug example**, not a
verified acquisition convention. The last flag is **debug-only**, not evidence
of verified semantics. Until
acquisition provenance is resolved, training marks the deployment manifest
`contract_verified: false`. TODO(agent): replace this debug gate with an audited
contract artifact after actual data/controller confirmation; do not flip the bit
merely to make an evaluation command pass.

Training fine-tunes pretrained DINOv3, freezes NeoBERT and trains the MeanFlow
head. Vision is BF16-autocast; head/JVP/objective stay FP32. It uses uniform chunk
sampling with replacement and explicit optimizer-step budgets, not alleged full
epochs. TensorBoard + JSONL record loss/gradient diagnostics; trusted `training.pt`
restores model/optimizer/scheduler and sampling/Torch RNG for the same episode and
step budget. A new run refuses to overwrite an existing output directory.

`policy_step_N.pt` is upstream `dem-v1`, with frozen towers included. The same-name
JSON stores SHA256 and protocol metadata. Local language/tokenizer assets must
remain available during loading. No success-rate claim follows from a decreasing
training loss; formal training still needs held-out episode/scenario validation.

## Unified evaluation (not yet GPU-validated)

```bash
bash starts/eval_dem_stack_cube.sh /absolute/path/policy_step_100.pt \
  --output-dir /absolute/path/eval-output
```

The primary YAML is
`examples/embodiment/config/isaaclab_franka_stack_cube_eval_dem_isaacsim6.yaml`.
It uses explicit GPU rank 0 for one rollout worker and one env worker, one
pipeline stage, seed 42, H=16/K=5/NFE=1, 450 control steps and one eval epoch.
The actor placement/model fields exist only for framework metadata; the evaluation
entrypoint does not launch an actor. A DEM deployment file goes through
`rollout.model.model_path` and `DEM.load`, **not** `runner.ckpt_path`.

An unverified manifest is refused by default. Debug-only override:
`actor.model.dem.allow_unverified_contract=true`; outputs remain experimental.

`env.eval.first_episode_only=true` is opt-in and leaves the existing pi05 configs'
window metrics unchanged. In this mode:

- `success_once`/`success_rate`/`policy_success` all select the first inner episode;
  `success_once_window` separately retains the legacy accumulation.
- Rewards, end step and terminated/truncated flags are frozen at the first end.
- Remaining chunk commands after done are replaced with zero relative motion/open
  gripper, never replayed as stale commands in a reset episode.
- Video stops at that episode. IsaacLab returns a **reset observation on an inner
  terminal step**, so that frame is omitted, not mislabeled as terminal; JSON
  explicitly reports `terminal_frame_available: false`. Outer-only timeout frames
  can be retained. Physics may continue for batch coordination after the first end,
  but those frames/rewards are not counted as another trial.

The existing launcher writes per-env JSON, `eval_results.json`,
`eval_throughput.json`, `eval_metrics.pt`, scenario JSONL and dual-camera MP4.
First-episode throughput uses first-success step for successes and first-episode
length for failures; it is not the old fixed-window failure budget. Compare pi05
and DEM only after applying the same metric, seed and horizon protocol to both.

## Checks

```bash
.venv-model312/bin/python -m pytest -q \
  tests/unit_tests/test_dem_stack_cube.py \
  tests/unit_tests/test_isaaclab_first_episode_success.py \
  tests/unit_tests/test_isaaclab_checkpoint_eval_script.py
bash -n starts/eval_dem_stack_cube.sh
```

The simulator-only probe is `toolkits/isaaclab/probe_dem_contract.py`; it requires
one approved GPU and the existing `starts/source.sh` environment. It never loads
DEM or claims policy success. Its attempted run failed before task reset.

Remaining acceptance gates: authorized backbone weights; successful real simulator
contract probe; actual MeanFlow parameter updates; real policy save/load equality
with fixed noise through the model factory; full first-episode video/JSON verification.
