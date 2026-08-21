#!/usr/bin/env python3
"""Merge local LeRobot dataset shards into one local dataset.

This script performs a file-level merge for locally generated LeRobot v2
datasets. It rewrites only the bookkeeping columns that must be globally unique
(`episode_index` and `index`) and copies all observation/action payload columns
without decoding images or depth arrays.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input LeRobot dataset roots.")
    parser.add_argument("--output-dir", required=True, help="Merged LeRobot dataset root.")
    parser.add_argument("--repo-id", required=True, help="Repo id metadata for the merged dataset.")
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_jsonl(path: Path):
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=True, indent=4)
        fp.write("\n")


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return int(episode_index) // int(chunks_size)


def _episode_parquet(root: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    rel = info["data_path"].format(
        episode_chunk=_episode_chunk(episode_index, chunks_size),
        episode_index=episode_index,
    )
    return root / rel


def _copy_episode_parquet(src: Path, dst: Path, new_episode_index: int, global_frame_start: int) -> int:
    table = pq.read_table(src)
    length = table.num_rows
    replacements = {
        "episode_index": pa.array([int(new_episode_index)] * length, type=pa.int64()),
        "index": pa.array(range(int(global_frame_start), int(global_frame_start) + length), type=pa.int64()),
    }
    for name, values in replacements.items():
        if name in table.column_names:
            table = table.set_column(table.schema.get_field_index(name), name, values)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    del table
    gc.collect()
    return length


def _renumber_episode_record(record: dict, new_episode_index: int) -> dict:
    updated = dict(record)
    updated["episode_index"] = int(new_episode_index)
    return updated


def _renumber_stats_record(record: dict, new_episode_index: int, global_frame_start: int, length: int) -> dict:
    updated = dict(record)
    updated["episode_index"] = int(new_episode_index)
    stats = updated.get("stats")
    if isinstance(stats, dict):
        if "episode_index" in stats:
            stats["episode_index"] = {
                "min": [int(new_episode_index)],
                "max": [int(new_episode_index)],
                "mean": [float(new_episode_index)],
                "std": [0.0],
                "count": [int(length)],
            }
        if "index" in stats:
            end = int(global_frame_start) + int(length) - 1
            stats["index"] = {
                "min": [int(global_frame_start)],
                "max": [int(end)],
                "mean": [float(global_frame_start + end) / 2.0],
                "std": stats["index"].get("std", [0.0]),
                "count": [int(length)],
            }
    return updated


def _merge_info(first_info: dict, input_infos: list[dict], repo_id: str, robot_type: str | None, fps: int | None) -> dict:
    info = dict(first_info)
    total_episodes = sum(int(item.get("total_episodes", 0)) for item in input_infos)
    total_frames = sum(int(item.get("total_frames", 0)) for item in input_infos)
    info["repo_id"] = repo_id
    info["robot_type"] = robot_type or first_info.get("robot_type", "franka_panda")
    info["fps"] = int(fps or first_info.get("fps", 20))
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(first_info.get("total_tasks", 1))
    info["total_chunks"] = int(_episode_chunk(max(total_episodes - 1, 0), int(first_info.get("chunks_size", 1000))) + 1)
    info["splits"] = {"train": f"0:{total_episodes}"}
    return info


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_roots = [Path(path).expanduser().resolve() for path in args.inputs]
    input_infos = [_load_json(root / "meta" / "info.json") for root in input_roots]
    first_info = input_infos[0]
    chunks_size = int(first_info.get("chunks_size", 1000))

    tasks = _load_jsonl(input_roots[0] / "meta" / "tasks.jsonl")
    episodes = []
    episodes_stats = []
    source_replay = []
    global_episode_index = 0
    global_frame_index = 0

    for root, info in zip(input_roots, input_infos, strict=True):
        shard_episodes = _load_jsonl(root / "meta" / "episodes.jsonl")
        shard_stats = _load_jsonl(root / "meta" / "episodes_stats.jsonl")
        shard_source = _load_jsonl(root / "meta" / "source_replay.jsonl")
        expected = int(info.get("total_episodes", len(shard_episodes)))
        if len(shard_episodes) != expected:
            raise ValueError(f"{root} has {len(shard_episodes)} episodes.jsonl rows, expected {expected}")

        for old_episode_index, episode in enumerate(shard_episodes):
            new_episode_index = global_episode_index
            length = int(episode["length"])
            print(
                json.dumps(
                    {
                        "mode": "merge_lerobot_fast_progress",
                        "input": str(root),
                        "old_episode_index": int(old_episode_index),
                        "new_episode_index": int(new_episode_index),
                        "length": int(length),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            src_parquet = _episode_parquet(root, info, old_episode_index)
            dst_rel = first_info["data_path"].format(
                episode_chunk=_episode_chunk(new_episode_index, chunks_size),
                episode_index=new_episode_index,
            )
            dst_parquet = output_dir / dst_rel
            written_length = _copy_episode_parquet(
                src_parquet,
                dst_parquet,
                new_episode_index=new_episode_index,
                global_frame_start=global_frame_index,
            )
            if written_length != length:
                raise ValueError(f"{src_parquet} has {written_length} rows, metadata says {length}")

            episodes.append(_renumber_episode_record(episode, new_episode_index))
            if old_episode_index < len(shard_stats):
                episodes_stats.append(
                    _renumber_stats_record(
                        shard_stats[old_episode_index],
                        new_episode_index,
                        global_frame_index,
                        length,
                    )
                )
            if old_episode_index < len(shard_source):
                record = dict(shard_source[old_episode_index])
                record["source_lerobot_root"] = str(root)
                record["source_lerobot_episode_index"] = int(old_episode_index)
                record["episode_index"] = int(new_episode_index)
                source_replay.append(record)

            global_episode_index += 1
            global_frame_index += length

    info = _merge_info(first_info, input_infos, args.repo_id, args.robot_type, args.fps)
    if int(info["total_frames"]) != global_frame_index:
        raise ValueError(f"merged frame count mismatch: {global_frame_index} vs {info['total_frames']}")

    _write_json(output_dir / "meta" / "info.json", info)
    _write_jsonl(output_dir / "meta" / "tasks.jsonl", tasks)
    _write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", episodes_stats)
    _write_jsonl(output_dir / "meta" / "source_replay.jsonl", source_replay)

    print(
        json.dumps(
            {
                "mode": "merge_lerobot_fast",
                "output_dir": str(output_dir),
                "inputs": [str(root) for root in input_roots],
                "episodes": int(global_episode_index),
                "frames": int(global_frame_index),
                "fps": int(info["fps"]),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
