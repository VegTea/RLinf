# Default-Environment LeRobot Replay For 600 Success Trajectories

For detailed handoff notes, verified output counts, merge details, visualization output, and known pitfalls, see `examples/embodiment/replay_lerobot_default600_handoff.md`.

This workflow replays the first 600 successful IsaacLab stack-cube trajectories and writes four local LeRobot datasets with 150 episodes each.

## Source

- Source trajectories: `/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_2k_traj_success_hit/success`
- Selection rule: lexicographic sort, then first `TOTAL_FILES=600`
- Input state: cube poses, cube orientations, joints, EEF state, and actions come from each `.npz`

## Environment

- GPUs: 8 processes, one process per GPU
- IsaacLab lifecycle: each GPU process creates IsaacLab once, then replays all trajectories assigned to that shard
- Table: fixed at `table_base.usd`
- Camera: default IsaacLab `wrist_cam` and `table_cam`
- Light: default IsaacLab lighting
- Visual scenario jsonl: disabled
- View scenario jsonl: disabled
- Light scenario jsonl: disabled

## Command

```bash
bash /inspire/hdd/global_user/gongjingjing-25039/xpyu/project/test/replay_success_lerobot_600_default_8gpu.sh
```

Common overrides:

```bash
RUN_TAG=my_default600_run \
RUN_SMOKE_PREFLIGHT=1 \
bash /inspire/hdd/global_user/gongjingjing-25039/xpyu/project/test/replay_success_lerobot_600_default_8gpu.sh
```

## Output

Run root:

```text
/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/datasets/isaaclab_success_lerobot_default600_runs/<RUN_TAG>
```

Important subdirectories:

- `source_shards/`: symlinked `.npz` files split across 8 shards
- `tmp_shards/`: one temporary LeRobot dataset per GPU shard
- `final/part_00` to `final/part_03`: final merged datasets, 150 episodes each
- `logs/`: smoke, replay, and merge logs

Each final dataset keeps provenance in:

```text
final/part_XX/meta/source_replay.jsonl
```

## Integrity Checks

The script refuses to merge if any shard writes fewer episodes than expected.

For the default 600-trajectory run:

- 8 shards
- 75 episodes per shard
- 2 shards per final part
- 150 episodes per final part

## Resume

For a failed run, inspect the failed shard log first:

```bash
less <RUN_ROOT>/logs/shard_XX/replay_lerobot.log
```

Then rerun the same `RUN_TAG` with:

```bash
RUN_TAG=<same_run_tag> \
LEROBOT_RESUME=1 \
bash /inspire/hdd/global_user/gongjingjing-25039/xpyu/project/test/replay_success_lerobot_600_default_8gpu.sh
```

The final merge overwrites `final/part_XX` after all shards have the expected episode count.
