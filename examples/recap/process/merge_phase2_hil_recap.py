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

"""Merge RSS phase2 HIL LeRobot datasets into task-level RECAP splits.

Phase2 is released as many complete LeRobot roots, one per team/model/task.
Episode indices and filenames overlap across those roots, so the merged
dataset cannot be represented by data-directory symlinks alone. This script
rewrites parquet files with contiguous episode/index columns and symlinks
videos to avoid copying large media files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DEFAULT_SOURCE_ROOT = Path(
    "/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/phase2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/recap/phase2"
)

TASK_PROMPTS = {
    "insert-mouse-battery": "Insert the battery into the mouse.",
    "seal-water-bottle-cap": (
        "Place the lid on the cup, align the threads, and twist clockwise to "
        "tighten."
    ),
    "tower-of-hanoi-game": (
        "Place the rings on the middle pillar under Tower of Hanoi constraints, "
        "ensuring the smaller ring ends up on top."
    ),
}

TASK_OUTPUT_NAMES = {
    "insert-mouse-battery": "insert_mouse_battery_hil_split",
    "seal-water-bottle-cap": "seal_water_bottle_cap_hil_split",
    "tower-of-hanoi-game": "tower_of_hanoi_game_hil_split",
}


@dataclass(frozen=True)
class EpisodeRecord:
    source_root: Path
    source_dataset: str
    source_episode_index: int
    source_file: Path
    length: int
    source_episode: dict[str, Any]
    source_stats: dict[str, Any] | None


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


def _format_lerobot_path(template: str, episode_index: int, chunks_size: int) -> str:
    chunk = _episode_chunk(episode_index, chunks_size)
    return template.format(
        chunk_index=chunk,
        episode_chunk=chunk,
        episode_index=episode_index,
    )


def _format_video_path(
    template: str, episode_index: int, chunks_size: int, video_key: str
) -> str:
    chunk = _episode_chunk(episode_index, chunks_size)
    return template.format(
        chunk_index=chunk,
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def _data_file(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return root / _format_lerobot_path(
        info["data_path"], episode_index, int(info.get("chunks_size", 1000))
    )


def _video_file(
    root: Path, info: dict[str, Any], episode_index: int, video_key: str
) -> Path:
    return root / _format_video_path(
        info["video_path"],
        episode_index,
        int(info.get("chunks_size", 1000)),
        video_key,
    )


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name not in table.column_names:
        return table.append_column(name, values)
    return table.set_column(table.schema.get_field_index(name), name, values)


def _task_from_dataset_name(dataset_name: str) -> str | None:
    for task_name in TASK_PROMPTS:
        if task_name in dataset_name:
            return task_name
    return None


def _source_roots_for_task(source_root: Path, task_name: str) -> list[Path]:
    roots = []
    for path in sorted(source_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if task_name not in path.name:
            continue
        if not path.name.startswith("HIL_"):
            continue
        if (path / "data").exists() and (path / "meta" / "info.json").exists():
            roots.append(path)
    return roots


def _build_parquet_map(source_root: Path) -> dict[int, Path]:
    parquet_by_episode = {}
    for parquet_file in sorted((source_root / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_file, columns=["episode_index"])
        if table.num_rows == 0:
            continue
        episode_index = int(table.column("episode_index")[0].as_py())
        if episode_index in parquet_by_episode:
            raise ValueError(
                f"Duplicate episode_index={episode_index} under {source_root}"
            )
        parquet_by_episode[episode_index] = parquet_file
    return parquet_by_episode


def _collect_episodes(source_roots: list[Path]) -> list[EpisodeRecord]:
    episodes = []
    for source_root in source_roots:
        info = _read_json(source_root / "meta" / "info.json")
        source_episodes = _read_jsonl(source_root / "meta" / "episodes.jsonl")
        parquet_by_episode = _build_parquet_map(source_root)
        stats_by_episode = {}
        stats_path = source_root / "meta" / "episodes_stats.jsonl"
        if stats_path.exists():
            stats_by_episode = {
                int(row["episode_index"]): row for row in _read_jsonl(stats_path)
            }

        for source_episode in sorted(
            source_episodes, key=lambda row: int(row["episode_index"])
        ):
            episode_index = int(source_episode["episode_index"])
            source_file = parquet_by_episode.get(episode_index)
            if source_file is None:
                source_file = _data_file(source_root, info, episode_index)
            if not source_file.exists():
                raise FileNotFoundError(
                    f"Missing parquet for episode {episode_index}: {source_file}"
                )
            length = int(source_episode.get("length", 0))
            if length <= 0:
                length = pq.ParquetFile(source_file).metadata.num_rows
            episodes.append(
                EpisodeRecord(
                    source_root=source_root,
                    source_dataset=source_root.name,
                    source_episode_index=episode_index,
                    source_file=source_file,
                    length=length,
                    source_episode=source_episode,
                    source_stats=stats_by_episode.get(episode_index),
                )
            )
    return episodes


def _split_episodes(
    episodes: list[EpisodeRecord], eval_ratio: float, seed: int
) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
    if not episodes:
        raise ValueError("No episodes found to split")
    eval_count = max(1, math.ceil(len(episodes) * eval_ratio))
    if eval_count >= len(episodes):
        raise ValueError(f"eval_ratio={eval_ratio} leaves no train episodes")

    positions = list(range(len(episodes)))
    rng = random.Random(seed)
    eval_positions = set(rng.sample(positions, eval_count))
    train = [episode for idx, episode in enumerate(episodes) if idx not in eval_positions]
    eval_ = [episode for idx, episode in enumerate(episodes) if idx in eval_positions]
    return train, eval_


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
    table = _replace_column(
        table,
        "index",
        pa.array(range(start_index, start_index + num_rows), type=pa.int64()),
    )
    table = _replace_column(
        table,
        "task_index",
        pa.array([0] * num_rows, type=pa.int64()),
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
    if "task_index" in stats:
        stats["task_index"] = {
            **stats["task_index"],
            "min": [0],
            "max": [0],
            "mean": [0.0],
        }
    row["stats"] = stats
    return row


def _copy_or_link_video(
    *,
    source_root: Path,
    output_root: Path,
    source_info: dict[str, Any],
    output_info: dict[str, Any],
    old_episode_index: int,
    new_episode_index: int,
    video_key: str,
) -> None:
    source_file = _video_file(source_root, source_info, old_episode_index, video_key)
    output_file = _video_file(output_root, output_info, new_episode_index, video_key)
    if not source_file.exists():
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() or output_file.is_symlink():
        output_file.unlink()
    output_file.symlink_to(source_file.resolve())


def _build_output_info(
    reference_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
) -> dict[str, Any]:
    info = dict(reference_info)
    info["robot_type"] = "yam"
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    return info


def _write_split(
    *,
    selected_episodes: list[EpisodeRecord],
    output_root: Path,
    reference_info: dict[str, Any],
    task_prompt: str,
) -> None:
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    total_frames = sum(episode.length for episode in selected_episodes)
    output_info = _build_output_info(
        reference_info,
        total_episodes=len(selected_episodes),
        total_frames=total_frames,
    )
    _write_json(output_root / "meta" / "info.json", output_info)
    _write_jsonl(
        output_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": task_prompt}],
    )

    output_episodes = []
    output_stats = []
    source_info_cache: dict[Path, dict[str, Any]] = {}
    running_index = 0

    progress = tqdm(
        enumerate(selected_episodes),
        total=len(selected_episodes),
        desc=f"Writing {output_root.name}",
    )
    for new_episode_index, episode in progress:
        output_file = _data_file(output_root, output_info, new_episode_index)
        length = _rewrite_episode_parquet(
            source_file=episode.source_file,
            output_file=output_file,
            new_episode_index=new_episode_index,
            start_index=running_index,
        )

        output_episodes.append(
            {
                "episode_index": new_episode_index,
                "tasks": [task_prompt],
                "length": length,
                "source_dataset": episode.source_dataset,
                "source_episode_index": episode.source_episode_index,
            }
        )

        if episode.source_stats is not None:
            output_stats.append(
                _rewrite_episode_stats(
                    episode.source_stats,
                    new_episode_index=new_episode_index,
                    length=length,
                    start_index=running_index,
                )
            )

        if episode.source_root not in source_info_cache:
            source_info_cache[episode.source_root] = _read_json(
                episode.source_root / "meta" / "info.json"
            )
        source_info = source_info_cache[episode.source_root]
        for video_key in _video_keys(output_info):
            _copy_or_link_video(
                source_root=episode.source_root,
                output_root=output_root,
                source_info=source_info,
                output_info=output_info,
                old_episode_index=episode.source_episode_index,
                new_episode_index=new_episode_index,
                video_key=video_key,
            )
        running_index += length

    _write_jsonl(output_root / "meta" / "episodes.jsonl", output_episodes)
    if output_stats:
        _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", output_stats)


def merge_task(
    *,
    source_root: Path,
    output_root: Path,
    task_name: str,
    eval_ratio: float,
    seed: int,
    force: bool,
    dry_run: bool,
) -> None:
    source_roots = _source_roots_for_task(source_root, task_name)
    if not source_roots:
        raise FileNotFoundError(f"No phase2 HIL source roots found for {task_name}")

    episodes = _collect_episodes(source_roots)
    train_episodes, eval_episodes = _split_episodes(episodes, eval_ratio, seed)

    task_output_root = output_root / TASK_OUTPUT_NAMES[task_name]
    print(f"\nTask: {task_name}")
    print(f"  Sources: {len(source_roots)}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Frames: {sum(episode.length for episode in episodes)}")
    print(f"  Train episodes: {len(train_episodes)}")
    print(f"  Eval episodes: {len(eval_episodes)}")
    print(f"  Output: {task_output_root}")
    for path in source_roots:
        print(f"    - {path.name}")

    if dry_run:
        return

    if task_output_root.exists():
        if not force:
            raise FileExistsError(f"{task_output_root} exists; pass --force")
        shutil.rmtree(task_output_root)

    reference_info = _read_json(source_roots[0] / "meta" / "info.json")
    _write_split(
        selected_episodes=train_episodes,
        output_root=task_output_root / "train",
        reference_info=reference_info,
        task_prompt=TASK_PROMPTS[task_name],
    )
    _write_split(
        selected_episodes=eval_episodes,
        output_root=task_output_root / "eval",
        reference_info=reference_info,
        task_prompt=TASK_PROMPTS[task_name],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--task",
        choices=tuple(TASK_PROMPTS) + ("all",),
        default="all",
        help="Task to merge, or all tasks.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.eval_ratio < 1.0):
        raise ValueError("--eval-ratio must be in (0, 1)")

    tasks = list(TASK_PROMPTS) if args.task == "all" else [args.task]
    for task_name in tasks:
        merge_task(
            source_root=args.source_root,
            output_root=args.output_root,
            task_name=task_name,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            force=args.force,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
