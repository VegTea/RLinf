# pi0.5-DROID wipe-board SFT

This directory contains the manual entry points for fine-tuning and evaluating
the official PyTorch `pi05_droid` checkpoint on the local wipe-board dataset.
Every script resolves the repository root from its own location, so it can be
launched from any working directory.

## Data contract

The SFT config uses the following mapping:

- `exterior_image_2_left` (right exterior camera) -> `base_0_rgb`
- `wrist_image_left` -> `left_wrist_0_rgb`
- zeros with a false mask -> `right_wrist_0_rgb`
- 7-D `joint_position` + 1-D `gripper_position` -> state
- 8-D absolute action targets -> 7 joint velocities at 15 Hz plus the unchanged
  absolute gripper target

Images embedded in parquet are decoded by the OpenPI LeRobot loader and then
resized with padding to 224 x 224. The local dataset contains 196 episodes and
145,011 frames at 15 Hz.

The default config uses seed 0 to hold out 20 complete episodes for validation
and trains on the remaining 176. For the current dataset this gives 130,650
training frames and 14,361 held-out frames. The split is written to
`episode_split.json` under the experiment directory. Each validation pass uses
10 evenly spaced, non-padded action chunks from every held-out episode. It
decodes the resulting 200 chunks with 10 flow steps and fixed sample-specific
noise. The logger records element-weighted MSE and MAE for normalized actions,
joint velocity, absolute gripper position, and integrated joint position under
`eval/`. Validation runs every 1,000 optimizer steps. Override
`runner.val_check_interval` to change the frequency; set it to `-1` to disable
periodic validation.

## 1. Train

The default is action-expert-only SFT on two GPUs and was smoke-tested for
three optimizer steps on two RTX 4090 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/sft/pi05_droid_wipe_board/run_sft.sh
```

To inspect one fully transformed batch before allocating the model, run:

```bash
source examples/sft/pi05_droid_wipe_board/common.sh
"$PYTHON_BIN" examples/sft/pi05_droid_wipe_board/validate_loader.py \
  --dataset-path "$DATASET_PATH" \
  --checkpoint-dir "$BASE_CHECKPOINT_DIR" \
  --norm-stats-dir "$NORM_STATS_DIR"
```

Additional Hydra overrides can be appended. For example, run 100 smoke-test
steps and save at step 100:

```bash
bash examples/sft/pi05_droid_wipe_board/run_sft.sh \
  runner.max_steps=100 runner.save_interval=100
```

For full-parameter SFT on a sufficiently large multi-GPU node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/sft/pi05_droid_wipe_board/run_sft.sh \
  cluster.component_placement.actor=0-3 \
  actor.model.openpi.train_expert_only=false \
  actor.global_batch_size=32
```

The default 10,000 optimizer steps are about 0.61 passes over the 176-episode
training split at global batch size 8. Checkpoints are written under
`outputs/pi05_droid_wipe_board_sft/droid_sft_openpi_pi05_wipe_board/checkpoints/`.
RLinf's OpenPI path currently does not inject LoRA adapters; `is_lora` only
affects FSDP wrapping and should not be treated as parameter-efficient tuning.

### One-H200 batch-size and learning-rate smoke test

Run the two-stage smoke test on exactly one NVIDIA H200:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash examples/sft/pi05_droid_wipe_board/run_h200_smoke_test.sh
```

Stage 1 tries micro batches `1 2 4 8 16` for three optimizer steps each. It
stops at the first OOM/failure, records the largest successful value, and uses
the next smaller candidate for the LR sweep as a memory safety margin. Stage 2
uses gradient accumulation 4 and compares `1e-5`, `2.5e-5`, and `5e-5` for 50
steps from the same base checkpoint and seed. The short trials use five warmup
steps followed by a constant LR; the regular 500-step cosine schedule would not
expose the requested LR during a 50-step smoke test.

Results are written under a timestamped directory in
`outputs/pi05_droid_h200_smoke/`:

- `max_stable_micro_batch.txt`
- `recommended_micro_batch.txt`
- `recommended_global_batch.txt`
- `recommended_lr_trial.txt`
- `summary.csv` and `summary.json`
- one TensorBoard directory and `train.log` per trial

The recommended LR trial is the finite run with the lowest mean loss over its
last ten updates. This is a smoke-test candidate, not a converged validation
result; confirm it with a longer held-out-episode experiment before the final
10,000-step SFT.

### Production training on four H100 GPUs

The production entry point uses the dual-4090 sweep result as a safe H100
starting point: micro batch 16 per GPU, global batch 64 without gradient
accumulation, peak learning rate `5e-5`, 500 warmup steps, and cosine decay.
It requires four visible H100 GPUs by default and records one-second GPU memory
and utilization samples in `gpu_metrics.csv`. At global batch 64, the default
10,000 steps process about 640,000 samples, or 4.9 passes over the 130,650-frame
training split:

First run the one-step smoke gate. It performs one optimizer update and then
10-step decoding on one fixed chunk from each held-out episode:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/sft/pi05_droid_wipe_board/run_h100_smoke.sh
```

After it prints `Smoke test complete`, start the full run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/sft/pi05_droid_wipe_board/run_h100_train.sh
```

Each run gets a timestamped directory under
`outputs/pi05_droid_h100_train/`. Override an environment default or append
Hydra overrides when necessary:

```bash
MAX_STEPS=20000 SAVE_INTERVAL=2000 LEARNING_RATE=2.5e-5 \
H100_TRAIN_ROOT=/path/to/output \
bash examples/sft/pi05_droid_wipe_board/run_h100_train.sh \
  actor.optim.weight_decay=1e-8
```

Set `VAL_CHECK_INTERVAL=500` to evaluate twice as often. One validation pass
performs 10-step decoding for the fixed 200-chunk set without updating
parameters. It preserves the training RNG stream so changing the validation
interval does not change later SFT noise samples.

This is an expert-only recipe. Re-run a batch probe on the target H100 node
before increasing the micro or global batch, and use held-out episodes or robot
success rate to select the final checkpoint.

### JAX-aligned RLinf implementation

The commands above use `model_type: openpi`, which wraps the official OpenPI
PyTorch model. For a closer implementation-level comparison with official
OpenPI JAX, use RLinf's local `openpi_pytorch` model (called `openpi_rlinf` in
the documentation). Its production recipe keeps the same split, batch size,
optimizer schedule, expert-only freeze policy, fixed evaluation noise, and
200-chunk validation protocol:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/sft/pi05_droid_wipe_board/run_openpi_pytorch_h100_smoke.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/sft/pi05_droid_wipe_board/run_openpi_pytorch_h100_train.sh
```

The aligned model expects a strict JAX-to-RLinf checkpoint conversion under
`PI05_DROID_RLINF_MODEL_PATH`; it must contain `config.json`,
`model.safetensors`, and the matching wipe-board normalization assets. Do not
label an `openpi` wrapper run as the aligned result.

After the runs finish, generate a machine-readable three-way comparison from
the OpenPI JAX JSON and the two RLinf TensorBoard directories:

```bash
python examples/sft/pi05_droid_wipe_board/compare_sft_evaluations.py \
  --jax-metrics /path/to/openpi/eval_metrics.json \
  --aligned-rlinf-run /path/to/openpi_pytorch_run \
  --wrapper-baseline-run /path/to/openpi_wrapper_run \
  --output /path/to/final_eval_comparison.json
```

RLinf records validation after optimizer update `N` at TensorBoard step
`N - 1`. The comparison tool maps it back to checkpoint step `N`, rejects
missing or non-finite metric rows, and keeps the aligned result and wrapper
baseline under distinct names.

Useful overrides:

```bash
# Change the search grid and trial length.
BATCH_CANDIDATES="2 4 8 12" \
LR_CANDIDATES="5e-6 1e-5 2.5e-5" \
LR_SWEEP_STEPS=100 \
bash examples/sft/pi05_droid_wipe_board/run_h200_smoke_test.sh

# Probe full-parameter SFT instead of the default action-expert-only regime.
TRAIN_EXPERT_ONLY=false \
bash examples/sft/pi05_droid_wipe_board/run_h200_smoke_test.sh
```

## 2. Export an SFT checkpoint

RLinf saves consolidated weights as `full_weights.pt`, while the official
OpenPI WebSocket loader expects `model.safetensors`. Export and validate the
key set and tensor shapes against the original `pi05_droid` checkpoint:

```bash
SFT_CHECKPOINT=outputs/pi05_droid_wipe_board_sft/\
droid_sft_openpi_pi05_wipe_board/checkpoints/global_step_10000 \
OUTPUT_CHECKPOINT_DIR=outputs/pi05_droid_wipe_board_sft/exported/step_10000 \
bash examples/sft/pi05_droid_wipe_board/export_sft_checkpoint.sh
```

For `openpi_pytorch` SFT checkpoints the exporter validates against the local
RLinf-layout `pytorch_rlinf` checkpoint, removes the SFT wrapper prefixes, and
writes BF16 bare-Pi0 weights. Override `REFERENCE_CHECKPOINT_DIR` only with a
checkpoint that has the same RLinf key layout.

The exporter bundles the wipe-board normalization statistics under
`assets/wipe_board_v1_zed196_force/norm_stats.json`. The RLinf server launcher
prefers this bundled copy, so an exported directory is self-contained. For an
older export without bundled stats, set `NORM_STATS_DIR` explicitly or let the
launcher fall back to the local `pytorch_rlinf` asset directory.

## 3. Start the WebSocket server

The default command serves the original converted checkpoint on port 8080:

```bash
bash examples/sft/pi05_droid_wipe_board/run_server.sh
```

The server opens its socket only after the model and normalization statistics
have loaded. To serve another OpenPI PyTorch export:

```bash
CHECKPOINT_DIR=/path/to/exported/openpi_checkpoint \
NORM_STATS_DIR=/path/to/wipe_board_norm_stats \
bash examples/sft/pi05_droid_wipe_board/run_server.sh
```

`CHECKPOINT_DIR` must contain `model.safetensors`; use the export step above for
an RLinf training checkpoint.

RLinf `openpi_pytorch` exports use a different key layout from the official
OpenPI PyTorch model. Start those exports with the dedicated launcher and select
the exterior view used during SFT:

```bash
CHECKPOINT_DIR=/path/to/exported/rlinf_checkpoint \
EXTERIOR_CAMERA=right \
SERVER_PORT=8080 \
bash examples/sft/pi05_droid_wipe_board/run_rlinf_server.sh

CHECKPOINT_DIR=/path/to/exported/rlinf_checkpoint \
EXTERIOR_CAMERA=left \
SERVER_PORT=8081 \
bash examples/sft/pi05_droid_wipe_board/run_rlinf_server.sh
```

The client sends only the selected exterior image and
`observation/wrist_image_left`; the policy creates the masked-zero third image
slot internally. The RLinf launcher loads BF16 weights, disables autograd for
the ten-step flow decode, and returns 15 absolute joint-position targets at
15 Hz.

By default `TORCH_COMPILE_DISABLE=1`, avoiding a long first-request compilation
on a 4090. Set it to `0` only when the service can be warmed up before use.

To debug the observations received during a real-robot rollout, enable the
server-side recorder:

```bash
OBSERVATION_RECORD_DIR="$PWD/outputs/pi05_droid_observation_debug" \
CHECKPOINT_DIR=/path/to/exported/rlinf_checkpoint \
EXTERIOR_CAMERA=right \
SERVER_PORT=8080 \
bash examples/sft/pi05_droid_wipe_board/run_rlinf_server.sh
```

Each server start creates a timestamped `session_*` directory. Every inference
request produces one line in `observations.jsonl`, including UTC, Unix, and
monotonic timestamps, all non-image observation values, and the corresponding
video frame index. Images are written as VS Code-compatible H.264 files:

- `exterior.mp4`: the selected raw exterior-camera stream.
- `wrist.mp4`: the raw wrist-camera stream.
- `model_inputs_224.mp4`: `[exterior, wrist, masked zero]` after OpenPI's
  224-by-224 resize-with-padding transform.

Stop the server with `Ctrl+C` so the MP4 files are finalized. The videos use
`OBSERVATION_RECORD_FPS=15` by default; override it only when the request rate
is different. Recording happens synchronously before inference and is intended
for debugging rather than maximum-throughput deployment. The server can only
record observations sent for policy inference: with DROID Infra
`exec_horizon=1`, this is every control step; with a larger execute horizon,
the intermediate open-loop control steps are not sent to the policy server.

## 4. Evaluate

In a second terminal, run:

```bash
bash examples/sft/pi05_droid_wipe_board/run_eval.sh
```

The script waits up to 900 seconds for a successful WebSocket metadata
handshake, so evaluation requests are never sent while the model is still
loading. It then samples 10 complete 15-step chunks from episode 0. Override
settings with environment variables:

```bash
SERVER_URL=ws://GPU_HOST:8080 EPISODE_INDEX=3 NUM_SAMPLES=20 \
bash examples/sft/pi05_droid_wipe_board/run_eval.sh
```

`run_eval.sh` defaults to `EXTERIOR_CAMERA=auto` and reads the required camera
from the WebSocket metadata. Set `EXTERIOR_CAMERA=left` or `right` to require a
specific view; a mismatch with the server is rejected before inference.

Results are saved under `outputs/pi05_droid_eval/` as JSON summaries and NPZ
arrays.

## Path overrides

The scripts recognize:

- `PYTHON_BIN`
- `WIPE_BOARD_DATASET_PATH`
- `PI05_DROID_MODEL_PATH`
- `PI05_DROID_NORM_STATS`
- `OPENPI_DATA_HOME`
- `CUDA_VISIBLE_DEVICES`

All proxy variables are cleared because the configured model, tokenizer cache,
normalization statistics, and dataset are local.
