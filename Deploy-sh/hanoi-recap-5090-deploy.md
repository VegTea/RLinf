# Hanoi RECAP Policy Server on RTX 5090

This note documents how to deploy the Hanoi RECAP policy server on another RTX
5090 workstation.

## Repository Layout

Keep the same paths as this workstation:

```bash
/home/user/Workspace/RLinf
/home/user/Workspace/openpi-xhc
```

The RECAP server runs from `RLinf/.venv`, but it imports OpenPI from the local
`openpi-xhc` checkout.

## Checkpoint

Place the checkpoint here:

```bash
/home/user/Workspace/RLinf/checkpoints/yam_tower_of_hanoi_game_step41000/yam_tower_of_hanoi_game_step41000/model.safetensors
```

The directory must also contain the checkpoint assets, especially:

```bash
/home/user/Workspace/RLinf/checkpoints/yam_tower_of_hanoi_game_step41000/yam_tower_of_hanoi_game_step41000/assets/tower-of-hanoi-game/expert-success-hil-suffix-mix-data
```

## Setup Environment

From `RLinf`:

```bash
cd /home/user/Workspace/RLinf
Deploy-sh/setup-5090-hanoi-recap-env.sh
```

Why this extra setup script is needed:

- RLinf official `uv sync` currently pins `torch==2.6.0`.
- RTX 5090 requires a PyTorch build with Blackwell / `sm_120` support.
- The script upgrades PyTorch to:

```text
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
```

It also pins the OpenPI runtime dependencies used by the RECAP server:

```text
jax==0.5.3
jaxlib==0.5.3
numpy==1.26.4
transformers==4.53.2
```

At the end it runs a small CUDA matmul test. Expected output includes:

```text
torch 2.8.0+cu128 cuda 12.8
device NVIDIA GeForce RTX 5090 (12, 0)
cuda_matmul_ok (1024, 1024)
```

## Start Server

Start the Hanoi RECAP policy server on port `8084`:

```bash
cd /home/user/Workspace/RLinf
Deploy-sh/start-hanoi-recap-8084.sh
```

This script uses:

```text
/home/user/Workspace/RLinf/.venv/bin/python
```

Default server parameters:

```text
host: 0.0.0.0
port: 8084
device: cuda:0
action_chunk: 50
action_dim: 14
guidance_scale: 0
```

Logs are written to:

```bash
/home/user/Workspace/RLinf/logs/recap_server/recap-8084.log
```

## Verify Server

From `openpi-xhc`, run the handshake test:

```bash
cd /home/user/Workspace/openpi-xhc
PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
uv run python third_party/policy_deployment/scripts/ping.py \
  --host 127.0.0.1 \
  --port 8084
```

Expected result:

```text
[PASS] handshake OK
```

The metadata should include:

```text
policy_name: RecapCfgPolicy
action_horizon: 50
action_dim: 14
state_dim: 14
guidance_scale: 0.0
```

## Run Compare Test

Use the Tower of Hanoi bundle:

```bash
cd /home/user/Workspace/openpi-xhc
PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=egl \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode compare \
  --bundle out/bundles/tower_episode_000000.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --host 127.0.0.1 \
  --port 8084 \
  --prompt "Place the rings on the middle pillar under Tower of Hanoi constraints, ensuring the smaller ring ends up on top." \
  --action-horizon 50 \
  --output out/openpi_compare_tower_recap_guidance0_8084.mp4 \
  --render-camera front \
  --fps 30
```

On this workstation, with `guidance_scale=0`, the Tower bundle result was:

```text
overall_MAE     = 0.005826
overall_RMSE    = 0.014286
overall_max_abs = 0.157285
```

## Common Issues

### CUDA error: no kernel image is available

This means the PyTorch wheel does not support RTX 5090 / `sm_120`.

Fix:

```bash
cd /home/user/Workspace/RLinf
.venv/bin/python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

Then verify:

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
print("ok")
PY
```

### Missing `openpi`

The server imports OpenPI from the local `openpi-xhc` checkout. Install it into
`RLinf/.venv`:

```bash
cd /home/user/Workspace/RLinf
uv pip install --python .venv/bin/python --no-deps -e /home/user/Workspace/openpi-xhc/packages/openpi-client
uv pip install --python .venv/bin/python --no-deps -e /home/user/Workspace/openpi-xhc
```

### Transformers import error

Use the OpenPI-compatible transformers version and copy the patched files:

```bash
cd /home/user/Workspace/RLinf
uv pip install --python .venv/bin/python transformers==4.53.2 tokenizers==0.21.4

pyver=$(.venv/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)

cp -r /home/user/Workspace/openpi-xhc/src/openpi/models_pytorch/transformers_replace/* \
  ".venv/lib/python${pyver}/site-packages/transformers/"
```

### Server starts but clients fail metadata ping

Use the `recap_policy_server.py` in this RLinf checkout. It includes compatibility
metadata for the existing `policy_deployment` websocket client:

```text
protocol_version
action_horizon
action_dim
state_dim
image_keys
image_shape
expects_prompt
accepts_compressed_images
```

### Guidance scale

For this Hanoi checkpoint, the best matching behavior was with conditional
guidance disabled:

```bash
--guidance-scale 0
```

In the model code, inference uses:

```text
v = (1 - guidance_scale) * v_uncond + guidance_scale * v_cond
```

So `guidance_scale=0` means only the unconditional branch is used.
