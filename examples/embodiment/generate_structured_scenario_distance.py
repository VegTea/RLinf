#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


DEFAULT_WEIGHTS = {
    "cube_pos": 0.35,
    "cube_layout": 0.25,
    "cube_rpy": 0.05,
    "camera_pos": 0.20,
    "camera_rot": 0.10,
    "table_asset": 0.05,
}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise ValueError(f"{path}:{line_no} is missing id")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fp.write("\n")


def vector(record: dict, key: str, length: int) -> list[float]:
    value = record.get(key)
    if value is None:
        raise ValueError(f"Scenario {record.get('id')} is missing {key}")
    if len(value) != length:
        raise ValueError(
            f"Scenario {record.get('id')} has invalid {key} length {len(value)}, "
            f"expected {length}"
        )
    return [float(item) for item in value]


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def wrap_angle(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def rpy_l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum(wrap_angle(x - y) ** 2 for x, y in zip(a, b)))


def normalize_quat(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 0.0:
        raise ValueError(f"Invalid zero quaternion: {q}")
    return [value / norm for value in q]


def quat_angle(a: list[float], b: list[float]) -> float:
    qa = normalize_quat(a)
    qb = normalize_quat(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    dot = max(-1.0, min(1.0, dot))
    if 1.0 - dot < 1e-12:
        return 0.0
    return 2.0 * math.acos(dot)


def cube_positions(record: dict) -> list[list[float]]:
    return [vector(record, f"cube_{idx}_pos", 3) for idx in range(1, 4)]


def cube_rpys(record: dict) -> list[list[float]]:
    return [vector(record, f"cube_{idx}_rpy", 3) for idx in range(1, 4)]


def pairwise_distances(points: list[list[float]]) -> list[float]:
    return [
        l2(points[0], points[1]),
        l2(points[0], points[2]),
        l2(points[1], points[2]),
    ]


def compute_raw_components(record: dict, reference: dict) -> dict[str, float]:
    cube_pos = cube_positions(record)
    ref_cube_pos = cube_positions(reference)
    cube_rpy = cube_rpys(record)
    ref_cube_rpy = cube_rpys(reference)

    cube_pos_distance = sum(
        l2(pos, ref_pos) for pos, ref_pos in zip(cube_pos, ref_cube_pos)
    ) / len(cube_pos)

    layout = pairwise_distances(cube_pos)
    ref_layout = pairwise_distances(ref_cube_pos)
    cube_layout_distance = sum(
        abs(value - ref_value) for value, ref_value in zip(layout, ref_layout)
    ) / len(layout)

    cube_rpy_distance = sum(
        rpy_l2(rpy, ref_rpy) for rpy, ref_rpy in zip(cube_rpy, ref_cube_rpy)
    ) / len(cube_rpy)

    camera_pos_distance = l2(
        vector(record, "table_cam_pos", 3),
        vector(reference, "table_cam_pos", 3),
    )
    camera_rot_distance = quat_angle(
        vector(record, "table_cam_rot", 4),
        vector(reference, "table_cam_rot", 4),
    )
    table_asset_distance = 0.0 if record.get("table_asset") == reference.get("table_asset") else 1.0

    return {
        "cube_pos": cube_pos_distance,
        "cube_layout": cube_layout_distance,
        "cube_rpy": cube_rpy_distance,
        "camera_pos": camera_pos_distance,
        "camera_rot": camera_rot_distance,
        "table_asset": table_asset_distance,
    }


def parse_weights(raw: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            key, value = item.split("=", maxsplit=1)
            key = key.strip()
            if key not in weights:
                raise ValueError(f"Unknown weight key {key!r}; valid keys: {sorted(weights)}")
            weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return {key: value / total for key, value in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute model-free structured distances from scenario parameters to "
            "one reference scenario."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--reference-id", default="009876")
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Comma-separated weights, e.g. "
            "cube_pos=0.35,cube_layout=0.25,cube_rpy=0.05,"
            "camera_pos=0.20,camera_rot=0.10,table_asset=0.05"
        ),
    )
    args = parser.parse_args()

    records = read_jsonl(args.source)
    by_id = {str(record["id"]): record for record in records}
    if args.reference_id not in by_id:
        raise ValueError(f"reference id {args.reference_id!r} not found in {args.source}")
    reference = by_id[args.reference_id]
    weights = parse_weights(args.weights)

    raw_by_id = {}
    maxima = {key: 0.0 for key in weights}
    for record in records:
        components = compute_raw_components(record, reference)
        raw_by_id[str(record["id"])] = components
        for key, value in components.items():
            maxima[key] = max(maxima[key], float(value))

    output_records = []
    for record in records:
        scenario_id = str(record["id"])
        components = raw_by_id[scenario_id]
        normalized = {
            key: (float(value) / maxima[key] if maxima[key] > 0.0 else 0.0)
            for key, value in components.items()
        }
        structured_distance = sum(weights[key] * normalized[key] for key in weights)
        output = dict(record)
        for key in weights:
            output[f"structured_{key}_distance_raw"] = components[key]
            output[f"structured_{key}_distance_norm"] = normalized[key]
        output["structured_distance"] = structured_distance
        output["structured_distance_reference_id"] = args.reference_id
        output_records.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_records)

    distances = [float(record["structured_distance"]) for record in output_records]
    sorted_records = sorted(
        output_records,
        key=lambda record: (float(record["structured_distance"]), str(record["id"])),
    )
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "reference_id": args.reference_id,
        "count": len(output_records),
        "distance_key": "structured_distance",
        "weights": weights,
        "normalization_maxima": maxima,
        "structured_distance_min": min(distances),
        "structured_distance_max": max(distances),
        "structured_distance_mean": sum(distances) / len(distances),
        "first_20_ids": [str(record["id"]) for record in sorted_records[:20]],
        "last_20_ids": [str(record["id"]) for record in sorted_records[-20:]],
    }
    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.with_name(args.output.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    print(
        "structured_distance="
        f"{summary['structured_distance_min']:.9f}.."
        f"{summary['structured_distance_max']:.9f}, "
        f"mean={summary['structured_distance_mean']:.9f}"
    )
    print("first_20_ids=" + ",".join(summary["first_20_ids"]))


if __name__ == "__main__":
    main()
