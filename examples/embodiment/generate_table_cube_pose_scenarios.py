#!/usr/bin/env python3
"""Generate table scenario JSONL with randomized non-overlapping cube poses."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_INPUT = SCRIPT_PATH.parent / "config/env/Isaaclab_table_test.jsonl"
DEFAULT_OUTPUT = SCRIPT_PATH.parent / "config/env/Isaaclab_table_cube_pose_test.jsonl"

DEFAULT_INITIAL_POSES = [
    ([0.4, 0.0, 0.0203], [0, 0, 0]),
    ([0.52, 0.08, 0.0203], [0, 0, 0]),
    ([0.6, -0.1, 0.0203], [0, 0, 0]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scenario JSONL with 50 table records and randomized cube poses."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.2, 0.66))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.15, 0.15))
    parser.add_argument("--z", type=float, default=0.0203)
    parser.add_argument("--min-separation", type=float, default=0.1)
    parser.add_argument("--max-sample-tries", type=int, default=5000)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise KeyError(f"Missing id at {path}:{line_no}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def xy_distance(a: list[float], b: list[float]) -> float:
    return math.dist(a[:2], b[:2])


def sample_pose_set(
    rng: random.Random,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z: float,
    min_separation: float,
    max_sample_tries: int,
) -> list[tuple[list[float], list[float]]]:
    poses: list[tuple[list[float], list[float]]] = []
    for cube_idx in range(3):
        for attempt in range(1, max_sample_tries + 1):
            pos = [
                rng.uniform(*x_range),
                rng.uniform(*y_range),
                z,
            ]
            rpy = [0, 0, rng.uniform(-math.pi, math.pi)]
            if all(xy_distance(pos, existing_pos) > min_separation for existing_pos, _ in poses):
                poses.append((pos, rpy))
                break
        else:
            raise RuntimeError(
                "Failed to sample non-overlapping cube pose set "
                f"after {max_sample_tries} tries for cube index {cube_idx}."
            )
    return poses


def round_float(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == -0.0 else rounded


def round_list(values: list[float]) -> list[float]:
    return [round_float(value) for value in values]


def set_cube_fields(record: dict, poses: list[tuple[list[float], list[float]]]) -> dict:
    updated = dict(record)
    for idx, (pos, rpy) in enumerate(poses, start=1):
        updated[f"cube_{idx}_pos"] = round_list(pos)
        updated[f"cube_{idx}_rpy"] = round_list(rpy)
    return updated


def validate_record(record: dict, min_separation: float):
    positions = [record[f"cube_{idx}_pos"] for idx in range(1, 4)]
    for i in range(3):
        for j in range(i + 1, 3):
            dist = xy_distance(positions[i], positions[j])
            if dist <= min_separation:
                raise ValueError(
                    f"Scenario {record.get('id')} cube_{i + 1}/cube_{j + 1} "
                    f"distance {dist:.6f} <= min_separation {min_separation}"
                )


def main():
    args = parse_args()
    records = load_jsonl(args.input)
    if len(records) < args.count:
        raise ValueError(f"Input has {len(records)} records, expected at least {args.count}")

    rng = random.Random(args.seed)
    generated = []
    for idx, record in enumerate(records[: args.count]):
        if idx == 0:
            poses = DEFAULT_INITIAL_POSES
        else:
            poses = sample_pose_set(
                rng=rng,
                x_range=tuple(args.x_range),
                y_range=tuple(args.y_range),
                z=args.z,
                min_separation=args.min_separation,
                max_sample_tries=args.max_sample_tries,
            )
        updated = set_cube_fields(record, poses)
        validate_record(updated, args.min_separation)
        generated.append(updated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        for record in generated:
            fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(
        f"Wrote {len(generated)} records to {args.output} "
        f"with seed={args.seed}, min_separation={args.min_separation}"
    )


if __name__ == "__main__":
    main()
