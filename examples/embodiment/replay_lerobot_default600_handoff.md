# Handoff: Default600 IsaacLab Replay To LeRobot

This document is the handoff note for the default-environment IsaacLab replay dataset work. It is meant for the next agent or developer who needs to inspect, reproduce, resume, merge, visualize, or modify this pipeline.

## Current Result

Completed run root:

```text
/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/success_lerobot_default600_8gpu_20260722_193226
```

Final LeRobot datasets:

```text
<RUN_ROOT>/final/part_00
<RUN_ROOT>/final/part_01
<RUN_ROOT>/final/part_02
<RUN_ROOT>/final/part_03
```

Each final part contains 150 successful replay episodes. The verified counts are:

```text
part_00: 150 episodes, 52928 frames
part_01: 150 episodes, 52747 frames
part_02: 150 episodes, 52994 frames
part_03: 150 episodes, 52601 frames
total:   600 episodes, 211270 frames
```

The final frame total matches the sum of the 8 temporary shard datasets.

## Data Semantics

Source trajectories:

```text
/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_2k_traj_success_hit/success
```

Selection rule for this run:

```text
find "$SRC" -type f -name '*.npz' | sort | head -n 600
```

Environment used for replay:

- Table: fixed `table_base.usd`
- Camera: IsaacLab default `wrist_cam` and `table_cam`
- Light: IsaacLab default light
- Cube pose/orientation: from each `.npz` trajectory
- Joint/eef/action replay: from each `.npz` trajectory
- No visual scenario jsonl
- No camera scenario jsonl
- No light scenario jsonl
- Cube color mapping: `{"red":"cube_2","blue":"cube_1","green":"cube_3"}`
- Task description: `Stack the red block on the blue block, then stack the green block on the red block.`

Final LeRobot keys include:

```text
observation.state
action
observation.images.wrist
observation.depths.wrist
observation.images.table
observation.depths.table
observation.joint_pos
observation.joint_vel
observation.eef_pos
observation.eef_quat
observation.gripper_pos
observation.cube_positions
observation.cube_orientations
observation.object
goal.step_index
goal.stage_id
timestamp
frame_index
episode_index
index
task_index
```

## Core Files

Generation script:

```text
/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/test/replay_success_lerobot_600_default_8gpu.sh
```

Replay implementation:

```text
examples/embodiment/replay_isaaclab_trajectory.py
rlinf/envs/isaaclab/replay_h5.py
rlinf/envs/isaaclab/replay_utils.py
```

Fast merge script used for the final parts:

```text
examples/embodiment/merge_lerobot_datasets.py
```

Visualization script used for the final parts:

```text
examples/embodiment/visualize_lerobot_parts.py
```

Short workflow doc:

```text
examples/embodiment/replay_lerobot_default600.md
```

## Run Layout

Under the completed run root:

```text
selected_success_npz.txt
source_shards/src_00 ... source_shards/src_07
tmp_shards/shard_00 ... tmp_shards/shard_07
final/part_00 ... final/part_03
logs/
visualization_20260723_125646/
```

Meaning:

- `selected_success_npz.txt`: exact selected `.npz` source list.
- `source_shards/`: symlinks to selected `.npz` files, split evenly across 8 shards.
- `tmp_shards/`: one local LeRobot dataset per GPU process. Each shard has 75 episodes.
- `final/`: merged datasets. Each `part_XX` merges two tmp shards, producing 150 episodes.
- `logs/shard_XX/replay_lerobot.log`: replay log for each GPU worker.
- `logs/part_XX_merge.log` and `logs/part_XX_merge_fast.log`: merge logs. Prefer the fast logs when present.
- `visualization_20260723_125646/`: sampled videos/contact sheets for manual inspection.

## Reproduce The Default600 Run

From the repo root:

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf

RUN_TAG=success_lerobot_default600_8gpu_$(date +'%Y%m%d_%H%M%S') \
RUN_SMOKE_PREFLIGHT=1 \
bash /inspire/hdd/global_user/gongjingjing-25039/xpyu/project/test/replay_success_lerobot_600_default_8gpu.sh
```

Important environment knobs:

```bash
TOTAL_FILES=600                 # must be divisible by GPU count and final part count
GPU_LIST="0 1 2 3 4 5 6 7"      # this script expects exactly 8 GPUs
FINAL_PARTS=4                   # 4 final datasets
TABLE_ASSET=table_base.usd
RUN_SMOKE_PREFLIGHT=1           # preflight all GPUs with IsaacLab smoke test
PREFLIGHT_SMOKE_TIMEOUT_S=1800
LEROBOT_RESUME=0
SKIP_MERGE=0
```

The script intentionally creates IsaacLab once per GPU process and replays all assigned trajectories in that same process. This avoids paying IsaacLab startup for every single `.npz`.

## Resume A Failed Run

First inspect failed shard logs:

```bash
RUN_ROOT=/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/<RUN_TAG>
less "$RUN_ROOT/logs/shard_XX/replay_lerobot.log"
```

Then rerun with the same `RUN_TAG`:

```bash
RUN_TAG=<same_run_tag> \
LEROBOT_RESUME=1 \
bash /inspire/hdd/global_user/gongjingjing-25039/xpyu/project/test/replay_success_lerobot_600_default_8gpu.sh
```

The script checks each `tmp_shards/shard_XX/meta/source_replay.jsonl` count before merging. If a shard has fewer than the expected 75 episodes, it exits before writing final merged parts.

## Merge Notes

The final successful merge used:

```text
examples/embodiment/merge_lerobot_datasets.py
```

This script performs a file-level parquet merge:

- reads source episode parquet files
- rewrites `episode_index`
- rewrites global `index`
- writes destination parquet files
- rewrites `meta/info.json`
- rewrites `meta/episodes.jsonl`
- rewrites `meta/episodes_stats.jsonl`
- rewrites `meta/source_replay.jsonl`

It does not decode and re-encode all images. This is much faster and less memory-heavy than constructing a new `LeRobotDataset` and calling `add_frame()` for every frame.

Manual merge command for one part:

```bash
RUN=/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/success_lerobot_default600_8gpu_20260722_193226

.venv/bin/python examples/embodiment/merge_lerobot_datasets.py \
  "$RUN/tmp_shards/shard_00" \
  "$RUN/tmp_shards/shard_01" \
  --output-dir "$RUN/final/part_00" \
  --repo-id isaaclab-stack-cube-default600-20260722_193226-part_00 \
  --robot-type franka_panda \
  --fps 20 \
  --overwrite
```

Avoid running many large merges in parallel on GPFS. Each episode parquet can be around hundreds of MB; parallel parquet rewrites can be IO-heavy and can also trigger memory pressure.

## Integrity Checks

Use this after generation or merge:

```bash
RUN=/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/success_lerobot_default600_8gpu_20260722_193226

python - <<'PY'
import json
from pathlib import Path
run = Path("/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/success_lerobot_default600_8gpu_20260722_193226")
for p in sorted((run / "final").glob("part_*")):
    info = json.load(open(p / "meta/info.json"))
    counts = {
        "episodes_jsonl": sum(1 for _ in open(p / "meta/episodes.jsonl")),
        "stats_jsonl": sum(1 for _ in open(p / "meta/episodes_stats.jsonl")),
        "source_jsonl": sum(1 for _ in open(p / "meta/source_replay.jsonl")),
        "parquet": len(list((p / "data").rglob("*.parquet"))),
    }
    print(p.name, info["total_episodes"], info["total_frames"], info["splits"], counts)
print("tmp_total_frames", sum(json.load(open(p / "meta/info.json"))["total_frames"] for p in sorted((run / "tmp_shards").glob("shard_*"))))
print("final_total_frames", sum(json.load(open(p / "meta/info.json"))["total_frames"] for p in sorted((run / "final").glob("part_*"))))
PY
```

Expected for the completed run:

```text
part_00 150 52928 {'train': '0:150'}
part_01 150 52747 {'train': '0:150'}
part_02 150 52994 {'train': '0:150'}
part_03 150 52601 {'train': '0:150'}
tmp_total_frames 211270
final_total_frames 211270
```

Optional LeRobot load check:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
root = Path("/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/success_lerobot_default600_8gpu_20260722_193226/final/part_00")
ds = LeRobotDataset(repo_id="local-check-part00", root=root, download_videos=False)
print(ds.meta.total_episodes, ds.num_frames, ds[0]["episode_index"], ds[len(ds)-1]["episode_index"])
PY
```

This scan can take a few minutes because LeRobot may regenerate/cache the train split from parquet.

## Visualization

Visualization generated for the completed run:

```text
/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/success_lerobot_default600_8gpu_20260722_193226/visualization_20260723_125646
```

Open:

```text
visualization_20260723_125646/index.html
```

It contains 16 sampled episodes:

- each part: episodes `0,50,100,149`
- each episode: one full MP4 and one 8-frame contact sheet
- MP4 layout: wrist view on the left, table view on the right

Regenerate the same style of visualization:

```bash
RUN=/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/success_lerobot_default600_8gpu_20260722_193226
OUT="$RUN/visualization_$(date +'%Y%m%d_%H%M%S')"

.venv/bin/python examples/embodiment/visualize_lerobot_parts.py \
  "$RUN/final/part_00" \
  "$RUN/final/part_01" \
  "$RUN/final/part_02" \
  "$RUN/final/part_03" \
  --output-dir "$OUT" \
  --episodes 0,50,100,last \
  --fps 20 \
  --sheet-frames 8
```

Manual spot checks already done:

- `part_00_episode_000000_sheet.jpg`
- `part_02_episode_000050_sheet.jpg`
- `part_03_episode_000149_sheet.jpg`

These showed the expected default gray table, wrist/table views, blue/red/green cubes, and stage progression.

## Important Pitfalls

1. Do not confuse `tmp_shards/` and `final/`.
   - `tmp_shards/shard_XX` are per-GPU intermediate datasets.
   - `final/part_XX` are the intended final 150-episode datasets.

2. Do not assume `total_videos > 0`.
   - This dataset stores images inside parquet as PNG bytes.
   - `meta/info.json` may report `total_videos: 0`.

3. Do not use the older slow merge path for large data unless necessary.
   - Creating a new `LeRobotDataset` and replaying `add_frame()` over every frame is slow and can leave incomplete metadata if interrupted.
   - Prefer `examples/embodiment/merge_lerobot_datasets.py`.

4. Always use `set -o pipefail` when piping Python output through `tee`.
   - Without `pipefail`, a failed Python command can be masked by a successful `tee`.

5. GPFS is close to full.
   - Around the merge time, `/inspire/hdd/global_user/gongjingjing-25039` was at about 97% usage.
   - Avoid duplicating these 100GB-scale datasets unnecessarily.

6. The dataset is not visually randomized.
   - This run intentionally uses default table, default camera, and default light.
   - It is different from prior 10k randomized table/camera/light replay runs.

## Related But Separate Pipelines

There are other scripts under `/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/test/`, including:

```text
replay_success_lerobot_8gpu.sh
replay_lerobot_distributed.sh
replay_lerobot_distributed_ori.sh
merge_lerobot_shards.py
visualize_lerobot_replay_dataset.py
```

Those are for the broader randomized replay/10k pipeline and recovery work. For this default600 dataset, prefer:

```text
replay_success_lerobot_600_default_8gpu.sh
examples/embodiment/merge_lerobot_datasets.py
examples/embodiment/visualize_lerobot_parts.py
```

## Modification Summary

Relevant changes made for this handoff:

- Added/updated default600 generation workflow.
- Added fast local LeRobot part merge script.
- Added final part visualization script.
- Verified the completed default600 run by metadata count, parquet count, frame count, and one actual LeRobotDataset load.
- Generated visualization artifacts for manual inspection.
