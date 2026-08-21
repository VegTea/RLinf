#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl"
REPO_PATH="${ROOT_DIR}/RLinf"
ISAAC_SETUP="${REPO_PATH}/isaac_sim/setup_conda_env.sh"

SCENARIO_FILE="${REPO_PATH}/rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl"
THRESHOLD="0.019309"
LIMIT="50"
OUTPUT_ROOT="${REPO_PATH}/logs/curriculum_stage0_renders"
OUTPUT_DIR=""
RENDER_FRAMES="6"
WIDTH="256"
HEIGHT="256"
SAVE_WRIST="0"
ISOLATION="app-reuse"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash examples/embodiment/render_stack_cube_curriculum_stage0.sh [options] [-- extra render args...]

Purpose:
  Render the first curriculum stage scenarios for manual inspection. By default
  this selects the 50 records with cosine_distance <= 0.019309 from the 10025
  scenario-distance JSONL, writes that subset into the output directory, and
  reuses render_stack_cube_all_scenarios.py to render table camera PNGs.

Options:
  --scenario-file PATH  Source JSONL with full scenario records and distance.
  --threshold FLOAT     Inclusive cosine_distance threshold. Default: 0.019309.
  --limit N             Expected/maximum selected records. Default: 50.
  --output-root DIR     Parent output directory. Default: RLinf/logs/curriculum_stage0_renders.
  --output-dir DIR      Exact output directory. Overrides --output-root timestamp.
  --render-frames N     Number of render frames before saving each image. Default: 6.
  --width N             Render width. Default: 256.
  --height N            Render height. Default: 256.
  --save-wrist          Also save wrist camera PNGs.
  --isolation MODE      Render isolation mode passed through to the renderer.
                        Default: app-reuse. Choices are renderer-defined.
  -h, --help            Show this help.

Examples:
  bash examples/embodiment/render_stack_cube_curriculum_stage0.sh

  bash examples/embodiment/render_stack_cube_curriculum_stage0.sh \
    --output-dir /tmp/curriculum_stage0_preview --save-wrist
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario-file)
      SCENARIO_FILE="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
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
    --render-frames)
      RENDER_FRAMES="$2"
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
    --save-wrist)
      SAVE_WRIST="1"
      shift
      ;;
    --isolation)
      ISOLATION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/$(date +'%Y%m%d-%H:%M:%S')-threshold-${THRESHOLD}-limit-${LIMIT}"
fi

mkdir -p "${OUTPUT_DIR}"
SUBSET_FILE="${OUTPUT_DIR}/curriculum_stage0_${LIMIT}.jsonl"
SUMMARY_FILE="${OUTPUT_DIR}/curriculum_stage0_summary.json"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
if [[ -f "${ISAAC_SETUP}" ]]; then
  set +u
  source "${ISAAC_SETUP}"
  set -u
fi

"${REPO_PATH}/.venv/bin/python" - <<PY
import json
from pathlib import Path

scenario_file = Path("${SCENARIO_FILE}")
subset_file = Path("${SUBSET_FILE}")
summary_file = Path("${SUMMARY_FILE}")
threshold = float("${THRESHOLD}")
limit = int("${LIMIT}")

records = []
with scenario_file.open("r", encoding="utf-8") as fp:
    for line_no, line in enumerate(fp, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "cosine_distance" not in record:
            raise KeyError(f"Missing cosine_distance at {scenario_file}:{line_no}")
        if float(record["cosine_distance"]) <= threshold:
            records.append(record)

records.sort(key=lambda item: (float(item["cosine_distance"]), str(item["id"])))
if len(records) < limit:
    raise RuntimeError(
        f"Only selected {len(records)} records with cosine_distance <= {threshold}, "
        f"expected at least {limit}."
    )
records = records[:limit]

subset_file.parent.mkdir(parents=True, exist_ok=True)
with subset_file.open("w", encoding="utf-8") as fp:
    for record in records:
        fp.write(json.dumps(record, ensure_ascii=False) + "\\n")

summary = {
    "source_scenario_file": str(scenario_file),
    "subset_file": str(subset_file),
    "threshold": threshold,
    "limit": limit,
    "selected_count": len(records),
    "min_cosine_distance": min(float(r["cosine_distance"]) for r in records),
    "max_cosine_distance": max(float(r["cosine_distance"]) for r in records),
    "scenario_ids": [str(r["id"]) for r in records],
}
summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY

RENDER_ARGS=(
  --scenario-file "${SUBSET_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --render-frames "${RENDER_FRAMES}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --isolation "${ISOLATION}"
)

if [[ "${SAVE_WRIST}" == "1" ]]; then
  RENDER_ARGS+=(--save-wrist)
fi

RENDER_ARGS+=("${EXTRA_ARGS[@]}")

echo "Rendering curriculum stage0 scenarios..."
echo "subset_file=${SUBSET_FILE}"
echo "summary_file=${SUMMARY_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "command=${REPO_PATH}/.venv/bin/python ${REPO_PATH}/examples/embodiment/render_stack_cube_all_scenarios.py ${RENDER_ARGS[*]}"

"${REPO_PATH}/.venv/bin/python" \
  "${REPO_PATH}/examples/embodiment/render_stack_cube_all_scenarios.py" \
  "${RENDER_ARGS[@]}"

echo "Saved curriculum stage0 renders to: ${OUTPUT_DIR}"
