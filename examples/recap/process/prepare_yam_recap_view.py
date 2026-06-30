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

"""Create RECAP-friendly LeRobot views for YAM RSS challenge data.

The raw RSS challenge tree usually uses three buckets: expert, success/HIL,
and failure. This script creates lightweight processed datasets with symlinked
``data`` and ``videos`` directories. It copies existing ``meta`` files when
available and only regenerates minimal ``meta`` files as a fallback. It never
modifies the raw data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_RAW_ROOT = Path(
    "/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw/"
    "insert-mouse-battery"
)
DEFAULT_OUTPUT_BASE = Path("data/recap")
DEFAULT_TASK = "Insert the battery into the mouse."
DEFAULT_TASK_NAME = "insert-mouse-battery"
REQUIRED_META_FILES = ("info.json", "tasks.jsonl", "episodes.jsonl")


def _default_output_root(task_name: str) -> Path:
    return DEFAULT_OUTPUT_BASE / task_name.replace("-", "_")


def _default_task_text(task_name: str) -> str:
    if task_name == DEFAULT_TASK_NAME:
        return DEFAULT_TASK
    return task_name.replace("-", " ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_first_task(source: Path, default_task: str) -> str:
    tasks_path = source / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return default_task
    with tasks_path.open() as f:
        for line in f:
            if line.strip():
                return str(json.loads(line).get("task", default_task))
    return default_task


def _episode_records(source: Path, task_name: str) -> tuple[list[dict[str, Any]], int]:
    records = []
    total_frames = 0
    for parquet_file in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_file, columns=["episode_index"])
        if table.num_rows == 0:
            continue
        episode_index = int(table.column("episode_index")[0].as_py())
        length = table.num_rows
        records.append(
            {
                "episode_index": episode_index,
                "tasks": task_name,
                "length": length,
            }
        )
        total_frames += length
    records.sort(key=lambda row: row["episode_index"])
    return records, total_frames


def _build_info(
    source: Path,
    reference_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
) -> dict[str, Any]:
    info_path = source / "meta" / "info.json"
    if info_path.exists():
        info = _read_json(info_path)
    else:
        info = dict(reference_info)

    info["codebase_version"] = info.get("codebase_version", "v2.1")
    info["robot_type"] = "yam"
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["chunks_size"] = info.get("chunks_size", 1000)
    info["fps"] = info.get("fps", reference_info.get("fps", 60))
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    info.setdefault("features", reference_info.get("features", {}))
    return info


def _replace_symlink(dst: Path, target: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(target.resolve(), target_is_directory=True)


def _has_complete_meta(source: Path) -> bool:
    meta_dir = source / "meta"
    return all((meta_dir / filename).exists() for filename in REQUIRED_META_FILES)


def _copy_existing_meta(source: Path, output: Path) -> None:
    shutil.copytree(source / "meta", output / "meta")


def prepare_view(
    *,
    name: str,
    source: Path,
    output: Path,
    reference_info: dict[str, Any],
    task_name: str,
    task: str,
    force: bool,
) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    output.mkdir(parents=True, exist_ok=True)

    _replace_symlink(output / "data", source / "data")
    if (source / "videos").exists():
        _replace_symlink(output / "videos", source / "videos")

    meta_dir = output / "meta"
    if meta_dir.exists():
        shutil.rmtree(meta_dir)

    if _has_complete_meta(source):
        _copy_existing_meta(source, output)
        episodes, total_frames = _episode_records(source, task_name)
        print(
            f"{name}: copied source meta, {len(episodes)} episodes, "
            f"{total_frames} frames -> {output}"
        )
        return

    meta_dir.mkdir(parents=True)

    episodes, total_frames = _episode_records(source, task_name)
    info = _build_info(
        source,
        reference_info=reference_info,
        total_episodes=len(episodes),
        total_frames=total_frames,
    )

    _write_json(meta_dir / "info.json", info)
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)

    src_stats = source / "meta" / "episodes_stats.jsonl"
    if src_stats.exists():
        shutil.copy2(src_stats, meta_dir / "episodes_stats.jsonl")

    print(f"{name}: {len(episodes)} episodes, {total_frames} frames -> {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Processed dataset root. Defaults to "
            "data/recap/<task-name-with-underscores>."
        ),
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Canonical task id stored in episodes.jsonl; defaults to raw-root name.",
    )
    parser.add_argument(
        "--default-task",
        default=None,
        help=(
            "Fallback language task when meta/tasks.jsonl is missing. Defaults "
            "to a task-name-derived string."
        ),
    )
    parser.add_argument("--expert-dir-name", default="expert-data")
    parser.add_argument("--success-dir-name", default="success-and-hil-data")
    parser.add_argument("--failure-dir-name", default="failure-data")
    parser.add_argument("--expert-output-name", default="expert")
    parser.add_argument("--success-output-name", default="success_hil")
    parser.add_argument("--failure-output-name", default="failure")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    task_name = args.task_name or args.raw_root.name
    output_root = args.output_root or _default_output_root(task_name)
    default_task = args.default_task or _default_task_text(task_name)

    expert_src = args.raw_root / args.expert_dir_name
    success_src = args.raw_root / args.success_dir_name
    failure_src = args.raw_root / args.failure_dir_name
    for source in (expert_src, success_src, failure_src):
        if not (source / "data").exists():
            raise FileNotFoundError(
                f"Missing LeRobot data directory: {source / 'data'}"
            )

    reference_info = _read_json(expert_src / "meta" / "info.json")
    expert_task = _read_first_task(expert_src, default_task)
    success_task = _read_first_task(success_src, expert_task)

    specs = [
        (args.expert_output_name, expert_src, expert_task),
        (args.success_output_name, success_src, success_task),
        (args.failure_output_name, failure_src, expert_task),
    ]
    print(f"raw_root: {args.raw_root}")
    print(f"output_root: {output_root}")
    print(f"task_name: {task_name}")
    for name, source, task in specs:
        prepare_view(
            name=name,
            source=source,
            output=output_root / name,
            reference_info=reference_info,
            task_name=task_name,
            task=task,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
