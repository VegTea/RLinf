#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"

CONFIG_NAME="${CONFIG_NAME:-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10_mix50_max300_32gpu}"
ROBOT_PLATFORM_ARG="${ROBOT_PLATFORM:-LIBERO}"

NNODES="${NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-300}"

export PYTHONWARNINGS="ignore::FutureWarning"
export NVIDIA_DRIVER_CAPABILITIES=all
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ISAAC_PATH="${ISAAC_PATH:-${REPO_PATH}/isaac_sim}"
export EXP_PATH="${EXP_PATH:-${ISAAC_PATH}/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_PATH}/kit}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM_ARG}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

detect_rank() {
  if [[ -n "${NODE_RANK:-}" ]]; then
    echo "${NODE_RANK}"
  elif [[ -n "${SLURM_NODEID:-}" ]]; then
    echo "${SLURM_NODEID}"
  elif [[ -n "${RANK:-}" && -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" == "${NNODES}" ]]; then
    echo "${RANK}"
  elif [[ -n "${OMPI_COMM_WORLD_NODE_RANK:-}" ]]; then
    echo "${OMPI_COMM_WORLD_NODE_RANK}"
  elif [[ -n "${RANK:-}" && -n "${LOCAL_RANK:-}" && "${GPUS_PER_NODE}" -gt 0 ]]; then
    echo $((RANK / GPUS_PER_NODE))
  else
    echo "0"
  fi
}

detect_head_addr() {
  if [[ -n "${HEAD_NODE:-}" ]]; then
    echo "${HEAD_NODE}"
  elif [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
  elif [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "${SLURM_NODELIST}" | head -n 1
  else
    hostname -I | awk '{print $1}'
  fi
}

detect_local_ip() {
  hostname -I | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ && $0 !~ /^127\\./ {print; exit}'
}

make_default_run_id() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "${SLURM_JOB_ID}-${CONFIG_NAME}"
  elif [[ -n "${MASTER_ADDR:-}" ]]; then
    printf '%s-%s\n' "${MASTER_ADDR}" "${CONFIG_NAME}" | tr '/: ' '---'
  else
    echo "$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
  fi
}

NODE_RANK_DETECTED="$(detect_rank)"
HEAD_NODE_DETECTED="$(detect_head_addr)"
NODE_IP_DETECTED="${NODE_IP:-$(detect_local_ip)}"
if [[ -z "${NODE_IP_DETECTED}" ]]; then
  echo "Failed to detect a non-loopback local node IP. Set NODE_IP explicitly." >&2
  exit 1
fi
LOG_ROOT="${REPO_PATH}/logs/ray_multinode"
RUN_ID="${RUN_ID:-$(make_default_run_id)}"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${LOG_DIR}"
HEAD_IP_FILE="${LOG_DIR}/head_ip.txt"

if [[ "${NODE_RANK_DETECTED}" == "0" ]]; then
  printf '%s\n' "${NODE_IP_DETECTED}" > "${HEAD_IP_FILE}"
  HEAD_RAY_IP="${NODE_IP_DETECTED}"
else
  deadline=$((SECONDS + RAY_START_TIMEOUT))
  until [[ -s "${HEAD_IP_FILE}" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for head IP file: ${HEAD_IP_FILE}" >&2
      exit 1
    fi
    sleep 2
  done
  HEAD_RAY_IP="$(head -n 1 "${HEAD_IP_FILE}")"
fi

RAY_ADDRESS="${HEAD_RAY_IP}:${RAY_PORT}"
NODE_LOG="${LOG_DIR}/node_${NODE_RANK_DETECTED}_$(hostname).log"

{
  echo "===== RLinf Ray multinode bootstrap ====="
  echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "hostname=$(hostname)"
  echo "node_rank=${NODE_RANK_DETECTED}"
  echo "nnodes=${NNODES}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "head_node=${HEAD_NODE_DETECTED}"
  echo "head_ray_ip=${HEAD_RAY_IP}"
  echo "node_ip=${NODE_IP_DETECTED}"
  echo "ray_address=${RAY_ADDRESS}"
  echo "config_name=${CONFIG_NAME}"
  echo "repo_path=${REPO_PATH}"
  echo "log_dir=${LOG_DIR}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "nvidia_visible_devices=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
  echo "slurm_job_id=${SLURM_JOB_ID:-<unset>}"
  echo "slurm_nodeid=${SLURM_NODEID:-<unset>}"
  echo "slurm_nodelist=${SLURM_NODELIST:-<unset>}"
  echo "master_addr=${MASTER_ADDR:-<unset>}"
  echo "rank=${RANK:-<unset>}"
  echo "world_size=${WORLD_SIZE:-<unset>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia_smi_gpu_count=$(nvidia-smi -L | wc -l)"
    nvidia-smi -L | sed 's/^/gpu: /'
  fi
  echo "===== selected environment variables ====="
  env | sort | grep -E '^(SLURM|MASTER|WORLD|RANK|LOCAL_RANK|NODE_RANK|OMPI|PMI|RAY|CUDA|NVIDIA|VK_|ISAAC|EXP_PATH|CARB_APP_PATH|ROBOT_PLATFORM)=' || true
  echo "===== end bootstrap parameters ====="
} | tee "${NODE_LOG}"

ray stop --force >>"${NODE_LOG}" 2>&1 || true
sleep 2

if [[ "${NODE_RANK_DETECTED}" == "0" ]]; then
  ray start \
    --head \
    --node-ip-address="${NODE_IP_DETECTED}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --block >>"${NODE_LOG}" 2>&1 &
else
  deadline=$((SECONDS + RAY_START_TIMEOUT))
  until ray start \
      --address="${RAY_ADDRESS}" \
      --node-ip-address="${NODE_IP_DETECTED}" \
      --num-gpus="${GPUS_PER_NODE}" >>"${NODE_LOG}" 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Failed to join Ray cluster at ${RAY_ADDRESS} within ${RAY_START_TIMEOUT}s." | tee -a "${NODE_LOG}"
      exit 1
    fi
    sleep 5
  done
fi

if [[ "${NODE_RANK_DETECTED}" == "0" ]]; then
  deadline=$((SECONDS + RAY_START_TIMEOUT))
  until ray status --address="${RAY_ADDRESS}" >>"${NODE_LOG}" 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Ray head did not become ready at ${RAY_ADDRESS} within ${RAY_START_TIMEOUT}s." | tee -a "${NODE_LOG}"
      exit 1
    fi
    sleep 3
  done

  export LOG_NAME_TAG="${LOG_NAME_TAG:-32gpu-${CONFIG_NAME}}"
  export RAY_ADDRESS="auto"
  echo "Starting RLinf driver on head node." | tee -a "${NODE_LOG}"
  set +e
  bash "${SCRIPT_DIR}/run_embodiment.sh" "${CONFIG_NAME}" "${ROBOT_PLATFORM_ARG}"
  status=$?
  set -e
  echo "RLinf driver exited with status ${status}; stopping Ray cluster." | tee -a "${NODE_LOG}"
  ray stop --force >>"${NODE_LOG}" 2>&1 || true
  exit "${status}"
else
  echo "Ray worker joined. Keeping node process alive while head runs training." | tee -a "${NODE_LOG}"
  while ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; do
    sleep 30
  done
  echo "Ray cluster is no longer reachable; worker node exits." | tee -a "${NODE_LOG}"
fi
