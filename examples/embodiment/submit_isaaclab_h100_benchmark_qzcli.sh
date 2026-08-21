#!/usr/bin/env bash

# Submit the one-node, two-H100 benchmark using the local qzcli resource cache.
# Dry-run is the default so an explicit --submit is required to allocate GPUs.
set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"

WORKSPACE="分布式训练空间"
PROJECT="高岳导师课程-具身智能机器人系统"
COMPUTE_GROUP="开发区-H100-cuda12.8版本-183核"
SPEC="26ef0d6e-330d-4650-a18a-7e1fbe8f3717"
IMAGE="${RLINF_13_IMAGE:-docker.sii.shaipower.online/inspire-studio/rlinf-xhc:1.3}"
NAME="isaaclab-nearest100-h100x2-benchmark"
PRIORITY="10"
SHM_GIB="256"
SUBMIT="false"
RUNTIME_GPUS="2"
INCLUDE_NODE=""
EXCLUDE_NODES=()
BENCHMARK_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash examples/embodiment/submit_isaaclab_h100_benchmark_qzcli.sh [options] [-- benchmark options]

Defaults target the cached resources for 高岳导师课程-具身智能机器人系统:
  workspace:     分布式训练空间
  compute group: 开发区-H100-cuda12.8版本-183核
  spec:          2x NVIDIA_H100_SXM_80G + 40 CPU cores + 400GB memory

The command is dry-run by default. Pass --submit only after qzcli login and
after checking the generated payload. The default image is the confirmed
RLinf 1.3 URI and can be overridden with RLINF_13_IMAGE or --image.

Options:
  --submit                 Create the qzcli task. Default is --dry-run.
  --image URI              Override RLINF_13_IMAGE.
  --name NAME              qzcli job name.
  --priority N             qzcli priority (default: 10, highest).
  --runtime-gpus N        GPUs exposed to Ray/benchmark inside the allocation (default: 2).
  --include-node NODE     Pin the allocation to a known-good node (optional).
  --exclude-node NODE     Exclude a known-bad node (repeatable, optional).
  --shm-gib N              Shared memory GiB (default: 256).
  --workspace NAME_OR_ID   Override workspace.
  --project NAME_OR_ID     Override project.
  --compute-group NAME_OR_ID
  --spec ID                Override GPU spec.
  --                       Arguments forwarded to benchmark_isaaclab_h100_scaling.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) SUBMIT="true"; shift ;;
        --image) IMAGE="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --priority) PRIORITY="$2"; shift 2 ;;
        --runtime-gpus) RUNTIME_GPUS="$2"; shift 2 ;;
        --include-node) INCLUDE_NODE="$2"; shift 2 ;;
        --exclude-node) EXCLUDE_NODES+=("$2"); shift 2 ;;
        --shm-gib) SHM_GIB="$2"; shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --project) PROJECT="$2"; shift 2 ;;
        --compute-group) COMPUTE_GROUP="$2"; shift 2 ;;
        --spec) SPEC="$2"; shift 2 ;;
        --) shift; BENCHMARK_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'." >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v qzcli >/dev/null 2>&1; then
    echo "ERROR: qzcli is not installed or not on PATH." >&2
    exit 2
fi
if ! [[ "${RUNTIME_GPUS}" =~ ^[1-9][0-9]*$ ]] || (( RUNTIME_GPUS > 2 )); then
    echo "ERROR: --runtime-gpus must be 1 or 2 for this allocation." >&2
    exit 2
fi

BENCHMARK_ARGS_QUOTED=""
if (( ${#BENCHMARK_ARGS[@]} > 0 )); then
    printf -v BENCHMARK_ARGS_QUOTED ' %q' "${BENCHMARK_ARGS[@]}"
fi
# qzcli executes the submitted command through /bin/sh. Keep the outer command
# POSIX-parseable: Bash's $'...' quoting (emitted by printf %q for newlines)
# causes an immediate syntax error in dash/ash before bash can be launched.
printf -v REPO_PATH_QUOTED '%q' "${REPO_PATH}"
VISIBLE_GPUS="0"
if (( RUNTIME_GPUS == 2 )); then
    VISIBLE_GPUS="0,1"
fi
INNER_COMMAND="set -euo pipefail; unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; trap 'ray stop --force || true' EXIT; cd ${REPO_PATH_QUOTED}; source .venv/bin/activate; export CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS}; nvidia-smi -L; python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == ${RUNTIME_GPUS}, torch.cuda.device_count(); print(torch.__version__)'; ray stop --force || true; ray start --head --num-gpus=${RUNTIME_GPUS} --disable-usage-stats; CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} bash examples/embodiment/benchmark_isaaclab_h100_scaling.sh${BENCHMARK_ARGS_QUOTED}"
printf -v QZ_COMMAND 'bash -lc %q' "${INNER_COMMAND}"

QZ_ARGS=(
    create
    --name "${NAME}"
    --workspace "${WORKSPACE}"
    --project "${PROJECT}"
    --compute-group "${COMPUTE_GROUP}"
    --spec "${SPEC}"
    --image "${IMAGE}"
    --instances 1
    --shm "${SHM_GIB}"
    --priority "${PRIORITY}"
    --command "${QZ_COMMAND}"
)

if [[ -n "${INCLUDE_NODE}" ]]; then
    QZ_ARGS+=(--include-node "${INCLUDE_NODE}")
fi
for node in "${EXCLUDE_NODES[@]}"; do
    QZ_ARGS+=(--exclude-node "${node}")
done

if [[ "${SUBMIT}" != "true" ]]; then
    QZ_ARGS+=(--dry-run)
fi

printf 'qzcli'
printf ' %q' "${QZ_ARGS[@]}"
printf '\n'
qzcli "${QZ_ARGS[@]}"
