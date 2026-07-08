#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/home/user/Workspace/RLinf}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/user/Workspace/openpi-xhc}"

RECAP_SESSION="${RECAP_SESSION:-recap-8084}"
HANOI_SESSION="${HANOI_SESSION:-hanoi-8085}"

RECAP_PORT="${RECAP_PORT:-8084}"
HANOI_PORT="${HANOI_PORT:-8085}"

RECAP_CHECKPOINT="${RECAP_CHECKPOINT:-${RLINF_ROOT}/checkpoints/yam_tower_of_hanoi_game_step41000/yam_tower_of_hanoi_game_step41000}"
RECAP_REPO_ID="${RECAP_REPO_ID:-assets/tower-of-hanoi-game/expert-success-hil-suffix-mix-data}"
RECAP_GUIDANCE_SCALE="${RECAP_GUIDANCE_SCALE:-0}"

HANOI_CHECKPOINT="${HANOI_CHECKPOINT:-checkpoints/hanoi_200k_assets_params}"
HANOI_CONFIG="${HANOI_CONFIG:-pi05_tower-of-hanoi-game_mixed}"

ACTION_CHUNK="${ACTION_CHUNK:-50}"
ACTION_DIM="${ACTION_DIM:-14}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RECAP_LOG_DIR="${RLINF_ROOT}/logs/recap_server"
HANOI_LOG_DIR="${RLINF_ROOT}/logs/openpi_hanoi"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|restart|status|start-recap|start-hanoi|stop-recap|stop-hanoi]

Starts policy servers from RLinf:
  recap: ${RECAP_SESSION} on port ${RECAP_PORT}, guidance_scale=${RECAP_GUIDANCE_SCALE}
  hanoi: ${HANOI_SESSION} on port ${HANOI_PORT}, checkpoint=${OPENPI_ROOT}/${HANOI_CHECKPOINT}

Useful overrides:
  RECAP_GUIDANCE_SCALE=1 $(basename "$0") restart
  CUDA_VISIBLE_DEVICES=0 $(basename "$0") start
  OPENPI_ROOT=/path/to/openpi-xhc $(basename "$0") start-hanoi
EOF
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

stop_session() {
  local session="$1"
  if tmux_has_session "${session}"; then
    tmux kill-session -t "${session}"
  fi
}

wait_for_port() {
  local port="$1"
  local name="$2"
  local deadline="${3:-120}"
  local elapsed=0
  while (( elapsed < deadline )); do
    if ss -ltnp | grep -Eq ":${port}\\b"; then
      echo "${name} is listening on ${port}"
      return 0
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done
  echo "Timed out waiting for ${name} on ${port}" >&2
  return 1
}

start_recap() {
  mkdir -p "${RECAP_LOG_DIR}"
  stop_session "${RECAP_SESSION}"
  : > "${RECAP_LOG_DIR}/recap-${RECAP_PORT}.log"

  tmux new-session -d -s "${RECAP_SESSION}" \
    "cd '${RLINF_ROOT}' && \
     PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' \
     .venv/bin/python toolkits/standalone_eval_scripts/openpi/recap_policy_server.py \
       --checkpoint-dir '${RECAP_CHECKPOINT}' \
       --config-name pi05_yam \
       --repo-id '${RECAP_REPO_ID}' \
       --host 0.0.0.0 \
       --port '${RECAP_PORT}' \
       --device cuda:0 \
       --action-chunk '${ACTION_CHUNK}' \
       --action-env-dim '${ACTION_DIM}' \
       --guidance-scale '${RECAP_GUIDANCE_SCALE}' \
       2>&1 | tee '${RECAP_LOG_DIR}/recap-${RECAP_PORT}.log'"
}

start_hanoi() {
  mkdir -p "${HANOI_LOG_DIR}"
  stop_session "${HANOI_SESSION}"
  : > "${HANOI_LOG_DIR}/hanoi-${HANOI_PORT}.log"

  tmux new-session -d -s "${HANOI_SESSION}" \
    "cd '${OPENPI_ROOT}' && \
     PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
     uv run scripts/serve_policy.py \
       --port='${HANOI_PORT}' \
       policy:checkpoint \
       --policy.config='${HANOI_CONFIG}' \
       --policy.dir='${HANOI_CHECKPOINT}' \
       2>&1 | tee '${HANOI_LOG_DIR}/hanoi-${HANOI_PORT}.log'"
}

status() {
  echo "tmux sessions:"
  tmux ls 2>/dev/null || true
  echo
  echo "listening ports:"
  ss -ltnp | grep -E ":(${RECAP_PORT}|${HANOI_PORT})\\b" || true
  echo
  echo "GPU processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

cmd="${1:-start}"
case "${cmd}" in
  start)
    start_hanoi
    wait_for_port "${HANOI_PORT}" "${HANOI_SESSION}" 120
    start_recap
    wait_for_port "${RECAP_PORT}" "${RECAP_SESSION}" 120
    status
    ;;
  stop)
    stop_session "${RECAP_SESSION}"
    stop_session "${HANOI_SESSION}"
    status
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    status
    ;;
  start-recap)
    start_recap
    wait_for_port "${RECAP_PORT}" "${RECAP_SESSION}" 120
    status
    ;;
  start-hanoi)
    start_hanoi
    wait_for_port "${HANOI_PORT}" "${HANOI_SESSION}" 120
    status
    ;;
  stop-recap)
    stop_session "${RECAP_SESSION}"
    status
    ;;
  stop-hanoi)
    stop_session "${HANOI_SESSION}"
    status
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
