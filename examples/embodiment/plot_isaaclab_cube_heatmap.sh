#!/usr/bin/env bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"

usage() {
    cat <<USAGE
Usage:
  bash examples/embodiment/plot_isaaclab_cube_heatmap.sh [LOG_DIR_OR_METRICS_PT] [extra plot args]

Examples:
  bash examples/embodiment/plot_isaaclab_cube_heatmap.sh
  bash examples/embodiment/plot_isaaclab_cube_heatmap.sh logs/20260421-10:16:20
  bash examples/embodiment/plot_isaaclab_cube_heatmap.sh logs/20260421-10:16:20/eval_metrics.pt --bins 40

Defaults:
  - uses the newest logs/*/eval_metrics.pt when no path is provided
  - plots all cubes
  - writes to <log_dir>/cube_heatmaps
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    INPUT_PATH="$1"
    shift
else
    INPUT_PATH="$(find "${REPO_PATH}/logs" -mindepth 2 -maxdepth 2 -name eval_metrics.pt -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "$INPUT_PATH" ]]; then
        echo "Cannot find eval_metrics.pt under ${REPO_PATH}/logs" >&2
        exit 1
    fi
fi

if [[ -d "$INPUT_PATH" ]]; then
    METRICS_PATH="${INPUT_PATH}/eval_metrics.pt"
else
    METRICS_PATH="$INPUT_PATH"
fi

if [[ ! -f "$METRICS_PATH" ]]; then
    echo "Cannot find metrics file: ${METRICS_PATH}" >&2
    exit 1
fi

HAS_CUBE_ARG=0
for arg in "$@"; do
    if [[ "$arg" == "--cube" || "$arg" == --cube=* ]]; then
        HAS_CUBE_ARG=1
        break
    fi
done

PYTHON_BIN="${PYTHON:-python}"
CMD=("$PYTHON_BIN" "${EMBODIED_PATH}/plot_isaaclab_cube_heatmap.py" "$METRICS_PATH")
if [[ "$HAS_CUBE_ARG" -eq 0 ]]; then
    CMD+=("--cube" "all")
fi
CMD+=("$@")

echo "${CMD[@]}"
"${CMD[@]}"
