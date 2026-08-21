#!/usr/bin/env python3
"""Generate scenario JSONLs from table/cube/camera parameter pools."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
RLINF_ROOT = SCRIPT_PATH.parents[2]
ALL_SETTING_DIR = RLINF_ROOT / "rlinf/assets_isaaclab/all_setting"
CONFIG_ENV_DIR = RLINF_ROOT / "examples/embodiment/config/env"

TABLE_PATH = ALL_SETTING_DIR / "color_tables.jsonl"
CUBE_PATH = ALL_SETTING_DIR / "cube_pose_50.jsonl"
CAMERA_PATH = ALL_SETTING_DIR / "table_cam_circle_50.jsonl"

ASSET_ROOT = RLINF_ROOT / "rlinf/assets_isaaclab/SeattleLabTable/color_tables"
ASSET_FALLBACK_ROOT = RLINF_ROOT / "rlinf/assets_isaaclab/SeattleLabTable"

DEFAULT_TABLE = {"table_asset": "table_base.usd"}
DEFAULT_CUBE = {
    "cube_1_pos": [0.4, 0.0, 0.0203],
    "cube_1_rpy": [0, 0, 0],
    "cube_2_pos": [0.52, 0.08, 0.0203],
    "cube_2_rpy": [0, 0, 0],
    "cube_3_pos": [0.60, -0.10, 0.0203],
    "cube_3_rpy": [0, 0, 0],
}
DEFAULT_CAMERA = {
    "table_cam_pos": [1.0, 0.0, 0.4],
    "table_cam_rot": [0.35355, -0.61237, -0.61237, 0.35355],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pairwise and 3-way IsaacLab scenario combinations."
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--pair-count", type=int, default=25)
    parser.add_argument("--triple-count", type=int, default=20)
    parser.add_argument(
        "--all-setting-output-dir",
        type=Path,
        default=ALL_SETTING_DIR / "combined_scenarios",
    )
    parser.add_argument(
        "--config-output-dir",
        type=Path,
        default=CONFIG_ENV_DIR / "combined_scenarios",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row:
                raise KeyError(f"Missing id at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def normalize_table(row: dict) -> dict:
    return {"source_table_id": row["id"], "table_asset": row["usd_file"]}


def normalize_cube(row: dict) -> dict:
    return {
        "source_cube_id": row["id"],
        "cube_1_pos": row["cube_1_pos"],
        "cube_1_rpy": row["cube_1_rpy"],
        "cube_2_pos": row["cube_2_pos"],
        "cube_2_rpy": row["cube_2_rpy"],
        "cube_3_pos": row["cube_3_pos"],
        "cube_3_rpy": row["cube_3_rpy"],
    }


def normalize_camera(row: dict) -> dict:
    return {
        "source_camera_id": row["id"],
        "table_cam_pos": row["table_cam_pos"],
        "table_cam_rot": row["table_cam_rot"],
    }


def find_default_index(rows: list[dict], predicate, label: str) -> int:
    matches = [idx for idx, row in enumerate(rows) if predicate(row)]
    if not matches:
        raise ValueError(f"Could not find default {label}")
    return matches[0]


def sample_with_required(rows: list[dict], count: int, required_idx: int, rng: random.Random) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"Cannot sample {count} rows from pool of {len(rows)}")
    indices = list(range(len(rows)))
    indices.remove(required_idx)
    chosen = [required_idx] + rng.sample(indices, count - 1)
    return [rows[idx] for idx in chosen]


def table_asset_exists(table_asset: str) -> bool:
    path = Path(table_asset)
    if path.is_absolute():
        return path.exists()
    return (ASSET_ROOT / table_asset).exists() or (ASSET_FALLBACK_ROOT / table_asset).exists()


def scenario_record(record_id: str, table: dict, cube: dict, camera: dict) -> dict:
    return {
        "id": record_id,
        "table_asset": table["table_asset"],
        "cube_1_pos": cube["cube_1_pos"],
        "cube_1_rpy": cube["cube_1_rpy"],
        "cube_2_pos": cube["cube_2_pos"],
        "cube_2_rpy": cube["cube_2_rpy"],
        "cube_3_pos": cube["cube_3_pos"],
        "cube_3_rpy": cube["cube_3_rpy"],
        "table_cam_pos": camera["table_cam_pos"],
        "table_cam_rot": camera["table_cam_rot"],
        "source_table_id": table.get("source_table_id", "default_table"),
        "source_cube_id": cube.get("source_cube_id", "default_cube"),
        "source_camera_id": camera.get("source_camera_id", "default_camera"),
    }


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def copy_outputs(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.jsonl")):
        shutil.copy2(src, dst_dir / src.name)


def build_pairwise(name: str, rows_a: list[dict], rows_b: list[dict], make_record):
    records = []
    for idx, (a, b) in enumerate(itertools.product(rows_a, rows_b), 1):
        records.append(make_record(f"{name}_{idx:04d}", a, b))
    return records


def build_triple(name: str, tables: list[dict], cubes: list[dict], cameras: list[dict]):
    records = []
    for idx, (table, cube, camera) in enumerate(itertools.product(tables, cubes, cameras), 1):
        records.append(scenario_record(f"{name}_{idx:05d}", table, cube, camera))
    return records


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    table_rows = [normalize_table(row) for row in load_jsonl(TABLE_PATH)]
    cube_rows = [normalize_cube(row) for row in load_jsonl(CUBE_PATH)]
    camera_rows = [normalize_camera(row) for row in load_jsonl(CAMERA_PATH)]

    default_table_idx = find_default_index(
        table_rows,
        lambda row: row["table_asset"] == DEFAULT_TABLE["table_asset"],
        "table",
    )
    default_cube_idx = find_default_index(
        cube_rows,
        lambda row: all(row[key] == value for key, value in DEFAULT_CUBE.items()),
        "cube",
    )
    default_camera_idx = find_default_index(
        camera_rows,
        lambda row: row["table_cam_pos"] == DEFAULT_CAMERA["table_cam_pos"],
        "camera",
    )

    pair_tables = sample_with_required(table_rows, args.pair_count, default_table_idx, rng)
    pair_cubes = sample_with_required(cube_rows, args.pair_count, default_cube_idx, rng)
    pair_cameras = sample_with_required(camera_rows, args.pair_count, default_camera_idx, rng)

    triple_tables = sample_with_required(table_rows, args.triple_count, default_table_idx, rng)
    triple_cubes = sample_with_required(cube_rows, args.triple_count, default_cube_idx, rng)
    triple_cameras = sample_with_required(camera_rows, args.triple_count, default_camera_idx, rng)

    for row in table_rows:
        if not table_asset_exists(row["table_asset"]):
            raise FileNotFoundError(f"Missing table asset: {row['table_asset']}")

    outputs = {
        "table_cube_25x25_625.jsonl": build_pairwise(
            "table_cube",
            pair_tables,
            pair_cubes,
            lambda record_id, table, cube: scenario_record(record_id, table, cube, DEFAULT_CAMERA),
        ),
        "table_camera_25x25_625.jsonl": build_pairwise(
            "table_camera",
            pair_tables,
            pair_cameras,
            lambda record_id, table, camera: scenario_record(record_id, table, DEFAULT_CUBE, camera),
        ),
        "cube_camera_25x25_625.jsonl": build_pairwise(
            "cube_camera",
            pair_cubes,
            pair_cameras,
            lambda record_id, cube, camera: scenario_record(record_id, DEFAULT_TABLE, cube, camera),
        ),
        "table_cube_camera_20x20x20_8000.jsonl": build_triple(
            "table_cube_camera",
            triple_tables,
            triple_cubes,
            triple_cameras,
        ),
    }

    args.all_setting_output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_jsonl(args.all_setting_output_dir / name, rows)
        print(f"wrote {len(rows)} rows: {args.all_setting_output_dir / name}")

    copy_outputs(args.all_setting_output_dir, args.config_output_dir)
    print(f"copied outputs to {args.config_output_dir}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
