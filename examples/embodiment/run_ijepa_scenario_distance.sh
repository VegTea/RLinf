#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:-/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf}"
IJEPA_REPO="${IJEPA_REPO:-/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/ijepa}"
ENV_DIR="${ENV_DIR:-${REPO_PATH}/.venv_ijepa_distance}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
PYTHON_BIN="${PYTHON_BIN:-}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
FORCE_INSTALL="${FORCE_INSTALL:-0}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
PYPI_EXTRA_INDEX_URL="${PYPI_EXTRA_INDEX_URL:-}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

CHECKPOINT="${CHECKPOINT:-${IJEPA_REPO}/checkpoints/IN1K-vit.h.14-300e.pth.tar}"
CHECKPOINT_URL="${CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar}"
DOWNLOAD_CHECKPOINT="${DOWNLOAD_CHECKPOINT:-0}"
CHECKPOINT_KEY="${CHECKPOINT_KEY:-encoder}"
MODEL_NAME="${MODEL_NAME:-vit_huge}"
PATCH_SIZE="${PATCH_SIZE:-14}"
CROP_SIZE="${CROP_SIZE:-224}"

SCENARIO_FILE="${SCENARIO_FILE:-${REPO_PATH}/rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-${REPO_PATH}/logs/20260629-17:54:39-全1wsetting初始图像/scenario_initial_renders}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_PATH}/rlinf/assets_isaaclab/all_setting/combined_with_ijepa_distance}"
STAGE_OUTPUT_DIR="${STAGE_OUTPUT_DIR:-${REPO_PATH}/rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_ijepa}"

REFERENCE_ID="${REFERENCE_ID:-009876}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda:0}"
if [[ -z "${IMAGE_PATTERN+x}" ]]; then
  IMAGE_PATTERN='scenario_{id}_table.png'
fi
COUNTS="${COUNTS:-50,100,200,400,600,800,989,1200,1500,1997,2500,3000,4048,5000,5964,7000,8014,9000,10025}"
DRY_RUN="${DRY_RUN:-0}"
NUM_RECORDS="${NUM_RECORDS:-}"

cd "${REPO_PATH}"

setup_python_env() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    echo "Using explicit PYTHON_BIN=${PYTHON_BIN}"
    return
  fi

  PYTHON_BIN="${ENV_DIR}/bin/python"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Creating i-JEPA venv: ${ENV_DIR}"
    "${BASE_PYTHON}" -m venv "${ENV_DIR}"
  fi

  if [[ "${SKIP_INSTALL}" == "1" ]]; then
    echo "SKIP_INSTALL=1, using ${PYTHON_BIN} without installing packages."
    return
  fi

  if [[ "${FORCE_INSTALL}" != "1" ]]; then
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
import torchvision
import numpy
import PIL
PY
    then
      echo "i-JEPA Python dependencies already available in ${ENV_DIR}."
      return
    fi
  fi

  echo "Installing i-JEPA dependencies into ${ENV_DIR}"
  pypi_args=(--index-url "${PYPI_INDEX_URL}")
  if [[ -n "${PYPI_EXTRA_INDEX_URL}" ]]; then
    pypi_args+=(--extra-index-url "${PYPI_EXTRA_INDEX_URL}")
  fi

  "${PYTHON_BIN}" -m pip install --upgrade pip "${pypi_args[@]}" ${PIP_EXTRA_ARGS}
  "${PYTHON_BIN}" -m pip install \
    torch torchvision \
    --index-url "${TORCH_CUDA_INDEX_URL}" \
    ${PIP_EXTRA_ARGS}
  "${PYTHON_BIN}" -m pip install \
    numpy pillow tqdm \
    "${pypi_args[@]}" \
    ${PIP_EXTRA_ARGS}
}

maybe_download_checkpoint() {
  if [[ -f "${CHECKPOINT}" ]]; then
    return
  fi
  if [[ "${DOWNLOAD_CHECKPOINT}" != "1" ]]; then
    echo "Missing i-JEPA checkpoint: ${CHECKPOINT}" >&2
    echo "Download manually with:" >&2
    echo "  mkdir -p \"$(dirname "${CHECKPOINT}")\"" >&2
    echo "  wget -O \"${CHECKPOINT}\" \"${CHECKPOINT_URL}\"" >&2
    echo "Or rerun with DOWNLOAD_CHECKPOINT=1." >&2
    exit 2
  fi

  mkdir -p "$(dirname "${CHECKPOINT}")"
  echo "Downloading i-JEPA checkpoint:"
  echo "  ${CHECKPOINT_URL}"
  echo "  -> ${CHECKPOINT}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${CHECKPOINT}" "${CHECKPOINT_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${CHECKPOINT_URL}" -o "${CHECKPOINT}"
  else
    echo "Neither wget nor curl is available for checkpoint download." >&2
    exit 2
  fi
}

setup_python_env
if [[ "${DRY_RUN}" != "1" ]]; then
  maybe_download_checkpoint
fi

echo "===== i-JEPA scenario distance ====="
echo "repo_path=${REPO_PATH}"
echo "ijepa_repo=${IJEPA_REPO}"
echo "env_dir=${ENV_DIR}"
echo "skip_install=${SKIP_INSTALL}"
echo "force_install=${FORCE_INSTALL}"
echo "torch_cuda_index_url=${TORCH_CUDA_INDEX_URL}"
echo "pypi_index_url=${PYPI_INDEX_URL}"
echo "pypi_extra_index_url=${PYPI_EXTRA_INDEX_URL}"
echo "python_bin=${PYTHON_BIN}"
echo "checkpoint=${CHECKPOINT}"
echo "checkpoint_key=${CHECKPOINT_KEY}"
echo "model_name=${MODEL_NAME}"
echo "patch_size=${PATCH_SIZE}"
echo "crop_size=${CROP_SIZE}"
echo "scenario_file=${SCENARIO_FILE}"
echo "image_dir=${IMAGE_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "stage_output_dir=${STAGE_OUTPUT_DIR}"
echo "reference_id=${REFERENCE_ID}"
echo "batch_size=${BATCH_SIZE}"
echo "device=${DEVICE}"
echo "counts=${COUNTS}"
echo "dry_run=${DRY_RUN}"
echo "===== start ====="

args=(
  "${REPO_PATH}/examples/embodiment/generate_ijepa_scenario_distance.py"
  --ijepa-repo "${IJEPA_REPO}"
  --checkpoint "${CHECKPOINT}"
  --checkpoint-key "${CHECKPOINT_KEY}"
  --model-name "${MODEL_NAME}"
  --patch-size "${PATCH_SIZE}"
  --crop-size "${CROP_SIZE}"
  --scenario-file "${SCENARIO_FILE}"
  --image-dir "${IMAGE_DIR}"
  --image-pattern "${IMAGE_PATTERN}"
  --output-dir "${OUTPUT_DIR}"
  --stage-output-dir "${STAGE_OUTPUT_DIR}"
  --stage-prefix "Isaaclab_19stage_10025_ijepa"
  --reference-id "${REFERENCE_ID}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --counts "${COUNTS}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ -n "${NUM_RECORDS}" ]]; then
  args+=(--num-records "${NUM_RECORDS}")
fi

"${PYTHON_BIN}" "${args[@]}" "$@"

echo "===== done ====="
echo "distance_jsonl=${OUTPUT_DIR}/Isaaclab_all_scenarios_10025_with_ijepa_distance.jsonl"
echo "summary=${OUTPUT_DIR}/ijepa_distance_summary.json"
echo "embeddings=${OUTPUT_DIR}/ijepa_embeddings_l2norm.npy"
echo "metadata=${OUTPUT_DIR}/ijepa_metadata.csv"
echo "stage_manifest=${STAGE_OUTPUT_DIR}/Isaaclab_19stage_10025_ijepa_stage_manifest.json"
