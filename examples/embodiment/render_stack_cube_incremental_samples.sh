#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl"
REPO_PATH="${ROOT_DIR}/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
RENDER_SCRIPT="${EMBODIED_PATH}/render_stack_cube_all_scenarios.py"
ISAAC_SETUP="${REPO_PATH}/isaac_sim/setup_conda_env.sh"

INPUT_DIR="${REPO_PATH}/logs/curriculum_range_eval/20260717-08:51:22-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum/scenario_sets/incremental"
OUTPUT_ROOT="${REPO_PATH}/logs/curriculum_incremental_sample_renders"
OUTPUT_DIR=""
SAMPLES_PER_STAGE="20"
SEED="42"
WIDTH="256"
HEIGHT="256"
RENDER_FRAMES="6"
ISOLATION="app-reuse"
SAVE_WRIST="0"
PREPARE_ONLY="0"
STAGE="all"
EXTRA_RENDER_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash examples/embodiment/render_stack_cube_incremental_samples.sh [options] [-- extra render args...]

Purpose:
  Render a fixed random sample of scenarios from each incremental curriculum stage.
  By default, it samples 20 scenarios from each stage_00.jsonl ... stage_18.jsonl
  and reuses the existing render_stack_cube_all_scenarios.py implementation.

Options:
  --input-dir DIR          Directory containing incremental/stage_XX.jsonl files.
  --output-root DIR        Parent output directory. Default: RLinf/logs/curriculum_incremental_sample_renders.
  --output-dir DIR         Exact output directory. Overrides --output-root timestamp.
  --samples-per-stage N    Number of scenarios to sample per stage. Default: 20.
  --seed N                 Fixed random seed. Default: 42.
  --stage N                Render one stage index, or all. Default: all.
  --width N                Render width. Default: 256.
  --height N               Render height. Default: 256.
  --render-frames N        Simulation render frames before saving image. Default: 6.
  --isolation MODE         shared, subprocess, fresh-env, app-reuse, or auto. Default: app-reuse.
  --save-wrist             Also save wrist camera images.
  --prepare-only           Only generate sampled jsonl files and summary, do not render.
  -h, --help               Show this help.

Examples:
  bash examples/embodiment/render_stack_cube_incremental_samples.sh

  bash examples/embodiment/render_stack_cube_incremental_samples.sh \
    --stage 0 --samples-per-stage 2 --prepare-only

  bash examples/embodiment/render_stack_cube_incremental_samples.sh --save-wrist -- --debug
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
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
    --samples-per-stage)
      SAMPLES_PER_STAGE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --render-frames)
      RENDER_FRAMES="$2"
      shift 2
      ;;
    --isolation)
      ISOLATION="$2"
      shift 2
      ;;
    --save-wrist)
      SAVE_WRIST="1"
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
      EXTRA_RENDER_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_RENDER_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${ISOLATION}" in
  auto|shared|subprocess|fresh-env|app-reuse) ;;
  *)
    echo "Unsupported --isolation ${ISOLATION}" >&2
    exit 2
    ;;
esac

if [[ "${STAGE}" != "all" && ! "${STAGE}" =~ ^[0-9]+$ ]]; then
  echo "Unsupported --stage ${STAGE}. Expected all or a non-negative integer." >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/$(date +'%Y%m%d-%H:%M:%S')-${SAMPLES_PER_STAGE}per-stage-seed-${SEED}"
fi

SAMPLED_DIR="${OUTPUT_DIR}/sampled_scenarios"
SUMMARY_JSON="${OUTPUT_DIR}/sample_summary.json"
RUN_LOG="${OUTPUT_DIR}/render_incremental_samples.log"

mkdir -p "${SAMPLED_DIR}"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
if [[ -f "${ISAAC_SETUP}" ]]; then
  set +u
  source "${ISAAC_SETUP}"
  set -u
fi

{
  echo "===== incremental sample render launch ====="
  echo "timestamp_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "input_dir=${INPUT_DIR}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "samples_per_stage=${SAMPLES_PER_STAGE}"
  echo "seed=${SEED}"
  echo "stage=${STAGE}"
  echo "width=${WIDTH}"
  echo "height=${HEIGHT}"
  echo "render_frames=${RENDER_FRAMES}"
  echo "isolation=${ISOLATION}"
  echo "save_wrist=${SAVE_WRIST}"
  echo "prepare_only=${PREPARE_ONLY}"
  echo "extra_render_args=${EXTRA_RENDER_ARGS[*]:-}"
  echo "===== end launch ====="
} | tee "${RUN_LOG}"

"${REPO_PATH}/.venv/bin/python" - <<PY
import json
import random
from pathlib import Path

input_dir = Path("${INPUT_DIR}")
sampled_dir = Path("${SAMPLED_DIR}")
summary_json = Path("${SUMMARY_JSON}")
samples_per_stage = int("${SAMPLES_PER_STAGE}")
seed = int("${SEED}")
stage_filter = "${STAGE}"

if samples_per_stage <= 0:
    raise ValueError(f"samples_per_stage must be positive, got {samples_per_stage}")
if not input_dir.exists():
    raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

stage_files = sorted(input_dir.glob("stage_*.jsonl"))
if stage_filter != "all":
    wanted = input_dir / f"stage_{int(stage_filter):02d}.jsonl"
    stage_files = [wanted]
if not stage_files:
    raise FileNotFoundError(f"No stage_*.jsonl files found under {input_dir}")

summary = {
    "input_dir": str(input_dir),
    "sampled_dir": str(sampled_dir),
    "samples_per_stage": samples_per_stage,
    "seed": seed,
    "stages": [],
}

for stage_file in stage_files:
    if not stage_file.exists():
        raise FileNotFoundError(f"stage file does not exist: {stage_file}")
    stage_name = stage_file.stem
    try:
        stage_index = int(stage_name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Unexpected stage file name: {stage_file}") from exc

    records = []
    with stage_file.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise KeyError(f"Missing id at {stage_file}:{line_no}")
            records.append(record)
    if not records:
        raise ValueError(f"No records in {stage_file}")

    rng = random.Random(seed + stage_index)
    sample_count = min(samples_per_stage, len(records))
    selected = rng.sample(records, sample_count)
    selected.sort(key=lambda item: str(item["id"]))

    stage_out_dir = sampled_dir / stage_name
    stage_out_dir.mkdir(parents=True, exist_ok=True)
    sampled_file = stage_out_dir / f"{stage_name}_sampled_{sample_count}.jsonl"
    with sampled_file.open("w", encoding="utf-8") as fp:
        for record in selected:
            fp.write(json.dumps(record, ensure_ascii=False) + "\\n")

    summary["stages"].append(
        {
            "stage_index": stage_index,
            "stage_name": stage_name,
            "source_file": str(stage_file),
            "source_count": len(records),
            "sample_count": sample_count,
            "sampled_file": str(sampled_file),
            "scenario_ids": [str(record["id"]) for record in selected],
            "render_dir": str(Path("${OUTPUT_DIR}") / stage_name / "renders"),
        }
    )

summary["stages"].sort(key=lambda item: item["stage_index"])
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(f"sample_summary={summary_json}")
for stage in summary["stages"]:
    print(
        f"{stage['stage_name']} source_count={stage['source_count']} "
        f"sample_count={stage['sample_count']} sampled_file={stage['sampled_file']}"
    )
PY

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Prepare-only mode enabled. Sampled scenario files are under: ${SAMPLED_DIR}" | tee -a "${RUN_LOG}"
  exit 0
fi

"${REPO_PATH}/.venv/bin/python" - <<PY | while IFS=$'\t' read -r stage_name sampled_file render_dir; do
import json
from pathlib import Path

summary = json.loads(Path("${SUMMARY_JSON}").read_text(encoding="utf-8"))
for stage in summary["stages"]:
    print("\\t".join([stage["stage_name"], stage["sampled_file"], stage["render_dir"]]))
PY
  echo "===== render ${stage_name} =====" | tee -a "${RUN_LOG}"
  CMD=(
    "${REPO_PATH}/.venv/bin/python"
    "${RENDER_SCRIPT}"
    --scenario-file "${sampled_file}"
    --output-dir "${render_dir}"
    --width "${WIDTH}"
    --height "${HEIGHT}"
    --render-frames "${RENDER_FRAMES}"
    --isolation "${ISOLATION}"
  )
  if [[ "${SAVE_WRIST}" == "1" ]]; then
    CMD+=(--save-wrist)
  fi
  CMD+=("${EXTRA_RENDER_ARGS[@]}")
  printf '%q ' "${CMD[@]}" | tee -a "${RUN_LOG}"
  echo | tee -a "${RUN_LOG}"
  "${CMD[@]}" 2>&1 | tee -a "${RUN_LOG}"
done

echo "Saved incremental sample renders to: ${OUTPUT_DIR}" | tee -a "${RUN_LOG}"
