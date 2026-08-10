#!/usr/bin/env bash

# Probe micro batch capacity and compare learning rates on one NVIDIA H200.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "This smoke test requires exactly one visible H200 GPU." >&2
  exit 1
fi
GPU_NAME="$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_MEMORY_MIB="$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
if [[ "${GPU_NAME}" != *H200* && "${ALLOW_NON_H200:-0}" != "1" ]]; then
  echo "Expected an NVIDIA H200, found: ${GPU_NAME}" >&2
  echo "Set ALLOW_NON_H200=1 only for script validation on another GPU." >&2
  exit 1
fi

BATCH_CANDIDATES="${BATCH_CANDIDATES:-1 2 4 8 16}"
LR_CANDIDATES="${LR_CANDIDATES:-1e-5 2.5e-5 5e-5}"
BATCH_PROBE_STEPS="${BATCH_PROBE_STEPS:-3}"
LR_SWEEP_STEPS="${LR_SWEEP_STEPS:-50}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
SMOKE_ROOT="${H200_SMOKE_ROOT:-${REPO_ROOT}/outputs/pi05_droid_h200_smoke/$(date -u +'%Y%m%d-%H%M%S')}"
mkdir -p "${SMOKE_ROOT}/batch_probe" "${SMOKE_ROOT}/lr_sweep"

cat > "${SMOKE_ROOT}/settings.txt" <<EOF
gpu_name=${GPU_NAME}
gpu_memory_mib=${GPU_MEMORY_MIB}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
batch_candidates=${BATCH_CANDIDATES}
lr_candidates=${LR_CANDIDATES}
batch_probe_steps=${BATCH_PROBE_STEPS}
lr_sweep_steps=${LR_SWEEP_STEPS}
lr_warmup_steps=${LR_WARMUP_STEPS}
gradient_accumulation=${GRADIENT_ACCUMULATION}
train_expert_only=${TRAIN_EXPERT_ONLY}
dataset=${DATASET_PATH}
checkpoint=${BASE_CHECKPOINT_DIR}
norm_stats=${NORM_STATS_DIR}
EOF

echo "H200 smoke root: ${SMOKE_ROOT}"
echo "GPU: ${GPU_NAME} (${GPU_MEMORY_MIB} MiB)"
echo "Training mode: train_expert_only=${TRAIN_EXPERT_ONLY}"

successful_batches=()
for micro_batch in ${BATCH_CANDIDATES}; do
  trial_dir="${SMOKE_ROOT}/batch_probe/mbs_${micro_batch}"
  echo "[batch probe] micro_batch=${micro_batch}, global_batch=${micro_batch}"
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    SFT_OUTPUT_ROOT="${trial_dir}" \
    bash "${SCRIPT_DIR}/run_sft.sh" \
      cluster.component_placement.actor=0-0 \
      runner.max_steps="${BATCH_PROBE_STEPS}" \
      runner.save_interval=-1 \
      runner.logger.experiment_name="h200_batch_mbs_${micro_batch}" \
      actor.micro_batch_size="${micro_batch}" \
      actor.global_batch_size="${micro_batch}" \
      actor.model.openpi.train_expert_only="${TRAIN_EXPERT_ONLY}" \
      actor.optim.lr_scheduler=constant \
      actor.optim.lr_warmup_steps=0; then
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_h200_smoke_test.py" \
      --validate-trial "${trial_dir}" --min-steps "${BATCH_PROBE_STEPS}"; then
      echo success > "${trial_dir}/status.txt"
      successful_batches+=("${micro_batch}")
      continue
    fi
  fi
  echo failed > "${trial_dir}/status.txt"
  echo "micro_batch=${micro_batch} failed; stopping the monotonic capacity probe."
  break
done

if (( ${#successful_batches[@]} == 0 )); then
  echo "No batch candidate completed successfully. See ${SMOKE_ROOT}/batch_probe." >&2
  exit 1
fi

max_index=$(( ${#successful_batches[@]} - 1 ))
max_stable_micro_batch="${successful_batches[${max_index}]}"
if [[ "${USE_MAX_BATCH:-0}" == "1" || ${#successful_batches[@]} -eq 1 ]]; then
  recommended_micro_batch="${max_stable_micro_batch}"
else
  recommended_index=$(( ${#successful_batches[@]} - 2 ))
  recommended_micro_batch="${successful_batches[${recommended_index}]}"
fi
recommended_global_batch=$(( recommended_micro_batch * GRADIENT_ACCUMULATION ))
echo "${max_stable_micro_batch}" > "${SMOKE_ROOT}/max_stable_micro_batch.txt"
echo "${recommended_micro_batch}" > "${SMOKE_ROOT}/recommended_micro_batch.txt"
echo "${recommended_global_batch}" > "${SMOKE_ROOT}/recommended_global_batch.txt"

echo "Maximum smoke-tested micro batch: ${max_stable_micro_batch}"
echo "LR sweep micro batch (one-step safety margin): ${recommended_micro_batch}"
echo "LR sweep global batch: ${recommended_global_batch}"

for learning_rate in ${LR_CANDIDATES}; do
  lr_name="${learning_rate//./p}"
  lr_name="${lr_name//-/_}"
  trial_dir="${SMOKE_ROOT}/lr_sweep/lr_${lr_name}"
  echo "[LR sweep] lr=${learning_rate}, micro=${recommended_micro_batch}, global=${recommended_global_batch}"
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    SFT_OUTPUT_ROOT="${trial_dir}" \
    bash "${SCRIPT_DIR}/run_sft.sh" \
      cluster.component_placement.actor=0-0 \
      runner.max_steps="${LR_SWEEP_STEPS}" \
      runner.save_interval=-1 \
      runner.logger.experiment_name="h200_lr_${lr_name}" \
      actor.micro_batch_size="${recommended_micro_batch}" \
      actor.global_batch_size="${recommended_global_batch}" \
      actor.model.openpi.train_expert_only="${TRAIN_EXPERT_ONLY}" \
      actor.optim.lr="${learning_rate}" \
      actor.optim.lr_scheduler=constant \
      actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}"; then
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_h200_smoke_test.py" \
      --validate-trial "${trial_dir}" --min-steps "${LR_SWEEP_STEPS}"; then
      echo success > "${trial_dir}/status.txt"
      continue
    fi
  fi
  echo failed > "${trial_dir}/status.txt"
  echo "Learning-rate trial ${learning_rate} failed; continuing with other candidates."
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_h200_smoke_test.py" --root "${SMOKE_ROOT}"
echo "Smoke test complete. Inspect ${SMOKE_ROOT}/summary.csv before the full run."
