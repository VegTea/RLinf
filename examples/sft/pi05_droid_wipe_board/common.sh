#!/usr/bin/env bash

# Shared paths and environment for the wipe-board pi0.5-DROID workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET_PATH="${DATASET_PATH:-${WIPE_BOARD_DATASET_PATH:-/inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force}}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-${PI05_DROID_MODEL_PATH:-/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/pi05_droid/pytorch}}"
NORM_STATS_DIR="${NORM_STATS_DIR:-${PI05_DROID_NORM_STATS:-${BASE_CHECKPOINT_DIR}/assets/wipe_board_v1_zed196_force}}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/inspire/hdd/global_user/czxs24230043/pretrained_models/PI/openpi_cache}"

export REPO_ROOT PYTHON_BIN DATASET_PATH BASE_CHECKPOINT_DIR NORM_STATS_DIR
export OPENPI_DATA_HOME
export WIPE_BOARD_DATASET_PATH="${DATASET_PATH}"
export PI05_DROID_MODEL_PATH="${BASE_CHECKPOINT_DIR}"
export PI05_DROID_NORM_STATS="${NORM_STATS_DIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# All default assets are local. Prevent an accidental proxy-routed download if
# a path is mistyped and OpenPI falls back to a remote URI.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
