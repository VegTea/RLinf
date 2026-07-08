#!/usr/bin/env bash
set -euo pipefail

cd /home/user/Workspace/RLinf

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11 .venv
fi

# Install RLinf first if the venv is new. RLinf's pyproject currently pins
# torch 2.6.0, so torch is upgraded to cu128 below for RTX 5090.
uv sync --inexact

# RTX 5090 requires a PyTorch build with Blackwell/sm_120 support.
.venv/bin/python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# OpenPI runtime pins used by the hanoi recap websocket server.
uv pip install --python .venv/bin/python \
  jax==0.5.3 jaxlib==0.5.3 numpy==1.26.4 \
  transformers==4.53.2 tokenizers==0.21.4 \
  safetensors websockets beartype==0.19.0 jaxtyping==0.2.36 \
  flax==0.10.2 orbax-checkpoint==0.11.13 sentencepiece chex==0.1.89 \
  dm-tree augmax flatbuffers 'fsspec[gcs]' pillow tyro tqdm-loggable rich polars \
  jsonlines datasets 'imageio[ffmpeg]' av pynput

# This workstation uses the local openpi-xhc checkout for OpenPI and
# openpi-client. Keep the same layout on another workstation for the fastest
# reproducible setup.
uv pip install --python .venv/bin/python --no-deps -e /home/user/Workspace/openpi-xhc/packages/openpi-client
uv pip install --python .venv/bin/python --no-deps -e /home/user/Workspace/openpi-xhc
uv pip install --python .venv/bin/python --no-deps \
  'git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'

pyver=$(.venv/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
cp -r /home/user/Workspace/openpi-xhc/src/openpi/models_pytorch/transformers_replace/* \
  ".venv/lib/python${pyver}/site-packages/transformers/"

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
print("cuda_matmul_ok", tuple(y.shape))
PY
