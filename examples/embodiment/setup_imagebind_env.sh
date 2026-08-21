#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf}"
IMAGEBIND_REPO="${IMAGEBIND_REPO:-/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/ImageBind}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-${IMAGEBIND_REPO}/.checkpoints/imagebind_huge.pth}"
DOWNLOAD_CHECKPOINT="${DOWNLOAD_CHECKPOINT:-0}"
INSTALL_FULL_IMAGEBIND_DEPS="${INSTALL_FULL_IMAGEBIND_DEPS:-0}"
PIP_INDEX_URL_ARG="${PIP_INDEX_URL_ARG:-https://pypi.org/simple}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -d "${IMAGEBIND_REPO}" ]]; then
  echo "ImageBind repo not found: ${IMAGEBIND_REPO}" >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install --index-url "${PIP_INDEX_URL_ARG}" ftfy iopath

if [[ "${INSTALL_FULL_IMAGEBIND_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install \
    --index-url "${PIP_INDEX_URL_ARG}" \
    --no-build-isolation \
    "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f" \
    -e "${IMAGEBIND_REPO}"
else
  "${PYTHON_BIN}" -m pip install \
    --index-url "${PIP_INDEX_URL_ARG}" \
    --no-build-isolation \
    "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f"
fi

if [[ "${DOWNLOAD_CHECKPOINT}" == "1" && ! -f "${CHECKPOINT}" ]]; then
  mkdir -p "$(dirname "${CHECKPOINT}")"
  "${PYTHON_BIN}" - <<PY
import torch
torch.hub.download_url_to_file(
    "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
    "${CHECKPOINT}",
    progress=True,
)
PY
fi

"${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${IMAGEBIND_REPO}")
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
print("ImageBind import OK")
print("ModalityType.VISION =", ModalityType.VISION)
PY

if [[ -f "${CHECKPOINT}" ]]; then
  ls -lh "${CHECKPOINT}"
else
  echo "Checkpoint not found yet: ${CHECKPOINT}"
  echo "Set DOWNLOAD_CHECKPOINT=1 to download it, or pass --checkpoint when running embedding."
fi
