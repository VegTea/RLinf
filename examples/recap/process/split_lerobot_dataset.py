# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split a local LeRobot dataset into train/eval episode views.

The output datasets are standalone LeRobot roots. Episode parquet files are
rewritten with contiguous episode indices, while videos are linked to the
source dataset to avoid copying large media files.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _data_file(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = _episode_chunk(episode_index, chunks_size)
    return root / info["data_path"].format(
        episode_chunk=episode_chunk,
        chunk_index=episode_chunk,
        episode_index=episode_index,
    )


def _video_file(
    root: Path, info: dict[str, Any], episode_index: int, video_key: str
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = _episode_chunk(episode_index, chunks_size)
    return root / info["video_path"].format(
        episode_chunk=episode_chunk,
        chunk_index=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name not in table.column_names:
        return table
    return table.set_column(table.schema.get_field_index(name), name, values)


def _rewrite_episode_parquet(
    source_file: Path,
    output_file: Path,
    new_episode_index: int,
    start_index: int,
) -> int:
    table = pq.read_table(source_file)
    num_rows = table.num_rows
    table = _replace_column(
        table,
        "episode_index",
        pa.array([new_episode_index] * num_rows, type=pa.int64()),
    )
    if "index" in table.column_names:
        table = _replace_column(
            table,
            "index",
            pa.array(range(start_index, start_index + num_rows), type=pa.int64()),
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_file)
    return num_rows


def _rewrite_episode_stats(
    row: dict[str, Any],
    new_episode_index: int,
    length: int,
    start_index: int,
) -> dict[str, Any]:
    row = dict(row)
    row["episode_index"] = new_episode_index
    stats = dict(row.get("stats", {}))
    if "episode_index" in stats:
        stats["episode_index"] = {
            **stats["episode_index"],
            "min": [new_episode_index],
            "max": [new_episode_index],
            "mean": [float(new_episode_index)],
        }
    if "index" in stats:
        stats["index"] = {
            **stats["index"],
            "min": [start_index],
            "max": [start_index + length - 1],
            "mean": [start_index + (length - 1) / 2.0],
        }
    row["stats"] = stats
    return row


def _copy_or_link_video(
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    old_episode_index: int,
    new_episode_index: int,
    video_key: str,
) -> None:
    source_file = _video_file(source_root, info, old_episode_index, video_key)
    output_file = _video_file(output_root, info, new_episode_index, video_key)
    if not source_file.exists():
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() or output_file.is_symlink():
        output_file.unlink()
    output_file.symlink_to(source_file.resolve())


def _write_split(
    *,
    source_root: Path,
    output_root: Path,
    selected_episodes: list[dict[str, Any]],
    stats_by_episode: dict[int, dict[str, Any]],
    info: dict[str, Any],
    force: bool,
) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} exists; pass --force to replace it")
        shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True)

    tasks_path = source_root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        shutil.copy2(tasks_path, output_root / "meta" / "tasks.jsonl")

    source_stats_path = source_root / "meta" / "stats.json"
    if source_stats_path.exists():
        shutil.copy2(source_stats_path, output_root / "meta" / "stats.json")

    output_info = dict(info)
    output_info["total_episodes"] = len(selected_episodes)
    output_info["total_frames"] = sum(int(row["length"]) for row in selected_episodes)
    output_info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    _write_json(output_root / "meta" / "info.json", output_info)

    output_episodes = []
    output_stats = []
    running_index = 0
    for new_episode_index, episode in enumerate(selected_episodes):
        old_episode_index = int(episode["episode_index"])
        source_file = _data_file(source_root, info, old_episode_index)
        output_file = _data_file(output_root, output_info, new_episode_index)
        length = _rewrite_episode_parquet(
            source_file=source_file,
            output_file=output_file,
            new_episode_index=new_episode_index,
            start_index=running_index,
        )

        output_episode = dict(episode)
        output_episode["episode_index"] = new_episode_index
        output_episode["length"] = length
        output_episodes.append(output_episode)

        if old_episode_index in stats_by_episode:
            output_stats.append(
                _rewrite_episode_stats(
                    stats_by_episode[old_episode_index],
                    new_episode_index=new_episode_index,
                    length=length,
                    start_index=running_index,
                )
            )

        for video_key in _video_keys(info):
            _copy_or_link_video(
                source_root=source_root,
                output_root=output_root,
                info=output_info,
                old_episode_index=old_episode_index,
                new_episode_index=new_episode_index,
                video_key=video_key,
            )
        running_index += length

    _write_jsonl(output_root / "meta" / "episodes.jsonl", output_episodes)
    if output_stats:
        _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", output_stats)


def split_dataset(
    *,
    source_root: Path,
    output_root: Path,
    eval_ratio: float,
    seed: int,
    mode: str,
    force: bool,
) -> tuple[Path, Path]:
    info = _read_json(source_root / "meta" / "info.json")
    episodes = _read_jsonl(source_root / "meta" / "episodes.jsonl")
    episodes = sorted(episodes, key=lambda row: int(row["episode_index"]))
    if not episodes:
        raise ValueError(f"No episodes found in {source_root}")

    stats_path = source_root / "meta" / "episodes_stats.jsonl"
    stats_by_episode = {}
    if stats_path.exists():
        for row in _read_jsonl(stats_path):
            stats_by_episode[int(row["episode_index"])] = row

    eval_count = max(1, math.ceil(len(episodes) * eval_ratio))
    if eval_count >= len(episodes):
        raise ValueError(
            f"eval_ratio={eval_ratio} leaves no train episodes for {source_root}"
        )

    if mode == "tail":
        eval_positions = set(range(len(episodes) - eval_count, len(episodes)))
    elif mode == "random":
        import random

        rng = random.Random(seed)
        eval_positions = set(rng.sample(range(len(episodes)), eval_count))
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    train_episodes = [
        episode for idx, episode in enumerate(episodes) if idx not in eval_positions
    ]
    eval_episodes = [
        episode for idx, episode in enumerate(episodes) if idx in eval_positions
    ]

    train_root = output_root / "train"
    eval_root = output_root / "eval"
    _write_split(
        source_root=source_root,
        output_root=train_root,
        selected_episodes=train_episodes,
        stats_by_episode=stats_by_episode,
        info=info,
        force=force,
    )
    _write_split(
        source_root=source_root,
        output_root=eval_root,
        selected_episodes=eval_episodes,
        stats_by_episode=stats_by_episode,
        info=info,
        force=force,
    )
    return train_root, eval_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("tail", "random"), default="tail")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_root, eval_root = split_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        mode=args.mode,
        force=args.force,
    )
    print(f"source_root: {args.source_root}")
    print(f"train_root: {train_root}")
    print(f"eval_root: {eval_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
