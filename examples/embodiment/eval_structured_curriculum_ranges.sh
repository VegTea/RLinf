#! /bin/bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl"
REPO_PATH="${ROOT_DIR}/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
ISAAC_SETUP="${REPO_PATH}/isaac_sim/setup_conda_env.sh"

CONFIG_NAME="isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum"
MODE="both"
STAGE="all"
OUTPUT_ROOT="${REPO_PATH}/logs/structured_curriculum_range_eval"
OUTPUT_DIR=""
STAGE_MANIFEST="${REPO_PATH}/rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_structured/Isaaclab_19stage_10025_structured_stage_manifest.json"
SCENARIO_SOURCE="${REPO_PATH}/rlinf/assets_isaaclab/all_setting/combined_with_structured_distance/Isaaclab_all_scenarios_10025_with_structured_distance.jsonl"
TOTAL_NUM_ENVS="32"
EVAL_ROLLOUT_EPOCH="20"
SKIP_EXISTING="0"
PREPARE_ONLY="0"
ROBOT_PLATFORM_ARG="${ROBOT_PLATFORM:-LIBERO}"
EXTRA_OVERRIDES=()

usage() {
  cat <<'EOF'
Usage:
  bash examples/embodiment/eval_structured_curriculum_ranges.sh [options] [-- extra hydra overrides...]

Purpose:
  Evaluate the original model on every structured-distance curriculum range in two modes:
    cumulative  - stage N contains all scenarios up to that stage
    incremental - stage N contains only scenarios newly added at that stage

Options:
  --config NAME              Hydra config name. Default: isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum.
  --mode MODE                both, cumulative, or incremental. Default: both.
  --stage N                  Run one stage index, or all. Default: all.
  --output-root DIR          Parent output directory. Default: RLinf/logs/structured_curriculum_range_eval.
  --output-dir DIR           Exact output directory. Overrides --output-root timestamp.
  --stage-manifest PATH      Curriculum stage manifest with incremental_ids/cumulative_ids.
  --scenario-source PATH     Full scenario JSONL with complete scenario records and structured_distance.
  --total-num-envs N         env.eval.total_num_envs override. Default: 32.
  --eval-rollout-epoch N     algorithm.eval_rollout_epoch override. Default: 20.
  --robot-platform NAME      ROBOT_PLATFORM value. Default: ${ROBOT_PLATFORM:-LIBERO}.
  --skip-existing            Skip a stage if its eval_metrics.pt already exists.
  --prepare-only             Generate scenario jsonl/metadata and exit before launching eval.
  -h, --help                 Show this help.

Examples:
  bash examples/embodiment/eval_structured_curriculum_ranges.sh

  bash examples/embodiment/eval_structured_curriculum_ranges.sh --mode incremental --stage 0 \
    --total-num-envs 4 --eval-rollout-epoch 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --stage-manifest)
      STAGE_MANIFEST="$2"
      shift 2
      ;;
    --scenario-source)
      SCENARIO_SOURCE="$2"
      shift 2
      ;;
    --total-num-envs)
      TOTAL_NUM_ENVS="$2"
      shift 2
      ;;
    --eval-rollout-epoch)
      EVAL_ROLLOUT_EPOCH="$2"
      shift 2
      ;;
    --robot-platform)
      ROBOT_PLATFORM_ARG="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING="1"
      shift
      ;;
    --prepare-only)
      PREPARE_ONLY="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

case "${MODE}" in
  both|cumulative|incremental) ;;
  *)
    echo "Unsupported --mode ${MODE}. Expected both, cumulative, or incremental." >&2
    exit 2
    ;;
esac

if [[ "${STAGE}" != "all" && ! "${STAGE}" =~ ^[0-9]+$ ]]; then
  echo "Unsupported --stage ${STAGE}. Expected all or a non-negative integer." >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
fi

RUNS_JSON="${OUTPUT_DIR}/runs.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
SUMMARY_CSV="${OUTPUT_DIR}/summary.csv"
MEGA_LOG_FILE="${OUTPUT_DIR}/eval_curriculum_ranges.log"

mkdir -p "${OUTPUT_DIR}"

export PYTHONWARNINGS="ignore::FutureWarning"
export EMBODIED_PATH
export REPO_PATH
export NVIDIA_DRIVER_CAPABILITIES=all
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export ISAAC_PATH="${ISAAC_PATH:-${REPO_PATH}/isaac_sim}"
export EXP_PATH="${EXP_PATH:-${ISAAC_PATH}/apps}"
export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_PATH}/kit}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
export DREAMZERO_PATH="${DREAMZERO_PATH:-/path/to/DreamZero}"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${DREAMZERO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export ROBOT_PLATFORM="${ROBOT_PLATFORM_ARG}"

if [[ -f "${ISAAC_SETUP}" ]]; then
  set +u
  source "${ISAAC_SETUP}"
  set -u
fi

{
  echo "===== curriculum range eval launch ====="
  echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "config_name=${CONFIG_NAME}"
  echo "mode=${MODE}"
  echo "stage=${STAGE}"
  echo "stage_manifest=${STAGE_MANIFEST}"
  echo "scenario_source=${SCENARIO_SOURCE}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "total_num_envs=${TOTAL_NUM_ENVS}"
  echo "eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}"
  echo "robot_platform=${ROBOT_PLATFORM}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "nvidia_visible_devices=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia_smi_gpu_count=$(nvidia-smi -L | wc -l)"
    nvidia-smi -L | sed 's/^/gpu: /'
  fi
echo "===== end launch ====="
} | tee "${MEGA_LOG_FILE}"

"${REPO_PATH}/.venv/bin/python" - <<PY
import json
from pathlib import Path

stage_manifest = Path("${STAGE_MANIFEST}")
scenario_source = Path("${SCENARIO_SOURCE}")
output_dir = Path("${OUTPUT_DIR}")
mode_filter = "${MODE}"
stage_filter = "${STAGE}"
runs_path = Path("${RUNS_JSON}")

manifest = json.loads(stage_manifest.read_text(encoding="utf-8"))
records_by_id = {}
with scenario_source.open("r", encoding="utf-8") as fp:
    for line_no, line in enumerate(fp, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        scenario_id = str(record["id"])
        records_by_id[scenario_id] = record

all_ids = []
for stage in manifest["stages"]:
    all_ids.extend(str(sid) for sid in stage["incremental_ids"])
missing = [sid for sid in all_ids if sid not in records_by_id]
if missing:
    raise KeyError(f"{len(missing)} scenario ids are missing from {scenario_source}: {missing[:10]}")

modes = ["cumulative", "incremental"] if mode_filter == "both" else [mode_filter]
selected_stage = None if stage_filter == "all" else int(stage_filter)
runs = []
scenario_sets_dir = output_dir / "scenario_sets"
for stage in manifest["stages"]:
    stage_index = int(stage["stage_index"])
    new_ids = [str(sid) for sid in stage["incremental_ids"]]
    cumulative_ids = [str(sid) for sid in stage["cumulative_ids"]]

    if selected_stage is not None and stage_index != selected_stage:
        continue

    for mode in modes:
        scenario_ids = cumulative_ids if mode == "cumulative" else new_ids
        stage_name = f"stage_{stage_index:02d}"
        scenario_file = scenario_sets_dir / mode / f"{stage_name}.jsonl"
        log_dir = output_dir / mode / stage_name
        scenario_file.parent.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        with scenario_file.open("w", encoding="utf-8") as fp:
            for scenario_id in scenario_ids:
                fp.write(json.dumps(records_by_id[scenario_id], ensure_ascii=False) + "\\n")
        metadata = {
            "mode": mode,
            "stage_index": stage_index,
            "range": (
                f"new structured stage {stage_index}"
                if mode == "incremental"
                else f"{manifest['distance_key']} <= {stage['threshold']}"
            ),
            "threshold": stage["threshold"],
            "previous_threshold": (
                manifest["stages"][stage_index - 1]["threshold"]
                if stage_index > 0
                else None
            ),
            "scenario_count": len(scenario_ids),
            "scenario_file": str(scenario_file),
            "log_dir": str(log_dir),
            "source_stage_manifest": str(stage_manifest),
            "source_scenario_file": str(scenario_source),
            "distance_key": manifest.get("distance_key", "structured_distance"),
            "incremental_min_distance": stage.get("incremental_min_distance"),
            "incremental_max_distance": stage.get("incremental_max_distance"),
        }
        (log_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        runs.append(metadata)

runs_path.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(f"Prepared {len(runs)} eval runs.")
print(f"runs_json={runs_path}")
PY

run_count="$("${REPO_PATH}/.venv/bin/python" - <<PY
import json
print(len(json.load(open("${RUNS_JSON}", "r", encoding="utf-8"))["runs"]))
PY
)"

if [[ "${run_count}" == "0" ]]; then
  echo "No runs selected." | tee -a "${MEGA_LOG_FILE}"
  exit 0
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Prepare-only mode enabled. Generated run metadata and scenario files under: ${OUTPUT_DIR}" | tee -a "${MEGA_LOG_FILE}"
  exit 0
fi

for ((run_idx=0; run_idx<run_count; run_idx++)); do
  run_info="$("${REPO_PATH}/.venv/bin/python" - <<PY
import json
run = json.load(open("${RUNS_JSON}", "r", encoding="utf-8"))["runs"][${run_idx}]
print("\\t".join([
    run["mode"],
    f"{int(run['stage_index']):02d}",
    run["scenario_file"],
    run["log_dir"],
    str(run["scenario_count"]),
]))
PY
)"
  IFS=$'\t' read -r run_mode stage_id scenario_file log_dir scenario_count <<< "${run_info}"
  eval_metrics_path="${log_dir}/eval_metrics.pt"
  eval_log_file="${log_dir}/eval_embodiment.log"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${eval_metrics_path}" ]]; then
    echo "[$((run_idx + 1))/${run_count}] skip existing ${run_mode}/stage_${stage_id} scenario_count=${scenario_count}" | tee -a "${MEGA_LOG_FILE}"
    continue
  fi

  echo "[$((run_idx + 1))/${run_count}] eval ${run_mode}/stage_${stage_id} scenario_count=${scenario_count}" | tee -a "${MEGA_LOG_FILE}"
  CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${log_dir}"
    "runner.only_eval=true"
    "runner.resume_dir=null"
    "runner.ckpt_path=null"
    "env.eval.total_num_envs=${TOTAL_NUM_ENVS}"
    "algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}"
    "env.eval.init_params.scenario_reset.enabled=true"
    "env.eval.init_params.scenario_reset.mode=random"
    "env.eval.init_params.scenario_reset.scenario_file=${scenario_file}"
    "env.eval.init_params.scenario_reset.curriculum.enabled=false"
    "${EXTRA_OVERRIDES[@]}"
  )
  printf '%q ' "${CMD[@]}" | tee -a "${MEGA_LOG_FILE}"
  echo | tee -a "${MEGA_LOG_FILE}"
  set +e
  (
    cd "${REPO_PATH}"
    "${CMD[@]}"
  ) 2>&1 | tee "${eval_log_file}"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" != "0" ]]; then
    echo "Eval failed for ${run_mode}/stage_${stage_id} with exit_status=${status}" | tee -a "${MEGA_LOG_FILE}"
    exit "${status}"
  fi
done

"${REPO_PATH}/.venv/bin/python" - <<PY
import csv
import json
from pathlib import Path

import torch

runs = json.load(open("${RUNS_JSON}", "r", encoding="utf-8"))["runs"]
summary_json = Path("${SUMMARY_JSON}")
summary_csv = Path("${SUMMARY_CSV}")
total_num_envs = int("${TOTAL_NUM_ENVS}")
eval_rollout_epoch = int("${EVAL_ROLLOUT_EPOCH}")
rows = []

def to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)

for run in runs:
    metrics_path = Path(run["log_dir"]) / "eval_metrics.pt"
    row = dict(run)
    row["eval_metrics_path"] = str(metrics_path)
    row["expected_sample_count"] = total_num_envs * eval_rollout_epoch
    row["status"] = "missing"
    if metrics_path.exists():
        data = torch.load(metrics_path, map_location="cpu", weights_only=False)
        aggregated = data.get("aggregated_metrics", {})
        row["status"] = "done"
        row["num_trajectories"] = int(aggregated.get("num_trajectories", 0) or 0)
        for key in ("success_once", "return", "episode_len", "reward"):
            row[key] = to_float(aggregated.get(key))
    rows.append(row)

summary = {
    "config_name": "${CONFIG_NAME}",
    "output_dir": "${OUTPUT_DIR}",
    "stage_manifest": "${STAGE_MANIFEST}",
    "scenario_source": "${SCENARIO_SOURCE}",
    "total_num_envs": total_num_envs,
    "eval_rollout_epoch": eval_rollout_epoch,
    "expected_sample_count_per_run": total_num_envs * eval_rollout_epoch,
    "runs": rows,
}
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

fieldnames = [
    "mode",
    "stage_index",
    "range",
    "threshold",
    "previous_threshold",
    "scenario_count",
    "expected_sample_count",
    "num_trajectories",
    "success_once",
    "return",
    "episode_len",
    "reward",
    "status",
    "log_dir",
    "scenario_file",
    "eval_metrics_path",
]
with summary_csv.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"summary_json={summary_json}")
print(f"summary_csv={summary_csv}")
PY

echo "Saved curriculum range eval outputs to: ${OUTPUT_DIR}" | tee -a "${MEGA_LOG_FILE}"
