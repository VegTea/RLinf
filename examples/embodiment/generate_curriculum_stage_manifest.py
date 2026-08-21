#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_COUNTS = "10,20,40,60,80,100,120,140,170,200"


def parse_counts(raw_counts: str) -> list[int]:
    counts = [int(value.strip()) for value in raw_counts.split(",") if value.strip()]
    if not counts:
        raise ValueError("counts must not be empty")
    if counts != sorted(counts):
        raise ValueError(f"counts must be sorted ascending: {counts}")
    if len(set(counts)) != len(counts):
        raise ValueError(f"counts must not contain duplicates: {counts}")
    return counts


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


def build_manifest(
    source_file: Path,
    selected_records: list[dict],
    counts: list[int],
    distance_key: str,
) -> dict:
    stages = []
    for stage_index, count in enumerate(counts):
        cumulative = selected_records[:count]
        previous_count = counts[stage_index - 1] if stage_index > 0 else 0
        incremental = selected_records[previous_count:count]
        distances = [float(record[distance_key]) for record in cumulative]
        incremental_distances = [float(record[distance_key]) for record in incremental]

        stages.append(
            {
                "stage_index": stage_index,
                "cumulative_count": len(cumulative),
                "incremental_count": len(incremental),
                "threshold": max(distances),
                "min_distance": min(distances),
                "max_distance": max(distances),
                "incremental_min_distance": min(incremental_distances),
                "incremental_max_distance": max(incremental_distances),
                "incremental_ids": [str(record["id"]) for record in incremental],
                "cumulative_ids": [str(record["id"]) for record in cumulative],
            }
        )

    return {
        "source_file": str(source_file),
        "distance_key": distance_key,
        "selection": "lowest_distance",
        "total_selected": len(selected_records),
        "cumulative_counts": counts,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a cumulative scenario curriculum manifest."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--distance-key", default="cosine_distance")
    parser.add_argument("--counts", default=DEFAULT_COUNTS)
    parser.add_argument(
        "--prefix",
        default="Isaaclab_10stage_200",
        help="Output file prefix.",
    )
    args = parser.parse_args()

    counts = parse_counts(args.counts)
    records = read_jsonl(args.source)
    missing_distance = [
        str(record["id"])
        for record in records
        if args.distance_key not in record
    ]
    if missing_distance:
        preview = ", ".join(missing_distance[:10])
        raise ValueError(
            f"{len(missing_distance)} records are missing {args.distance_key}: {preview}"
        )
    if counts[-1] > len(records):
        raise ValueError(
            f"largest count {counts[-1]} exceeds available records {len(records)}"
        )

    sorted_records = sorted(
        records,
        key=lambda record: (float(record[args.distance_key]), str(record["id"])),
    )
    selected_records = sorted_records[: counts[-1]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_jsonl = args.output_dir / f"{args.prefix}_with_distance.jsonl"
    manifest_json = args.output_dir / f"{args.prefix}_stage_manifest.json"
    write_jsonl(selected_jsonl, selected_records)

    manifest = build_manifest(
        source_file=args.source,
        selected_records=selected_records,
        counts=counts,
        distance_key=args.distance_key,
    )
    manifest["scenario_file"] = str(selected_jsonl)
    with manifest_json.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    stages_dir = args.output_dir / "stage_sets"
    incremental_dir = stages_dir / "incremental"
    cumulative_dir = stages_dir / "cumulative"
    incremental_dir.mkdir(parents=True, exist_ok=True)
    cumulative_dir.mkdir(parents=True, exist_ok=True)
    records_by_id = {str(record["id"]): record for record in selected_records}
    for stage in manifest["stages"]:
        stage_index = int(stage["stage_index"])
        write_jsonl(
            incremental_dir / f"stage_{stage_index:02d}.jsonl",
            [records_by_id[scenario_id] for scenario_id in stage["incremental_ids"]],
        )
        write_jsonl(
            cumulative_dir / f"stage_{stage_index:02d}.jsonl",
            [records_by_id[scenario_id] for scenario_id in stage["cumulative_ids"]],
        )

    print(f"wrote {selected_jsonl}")
    print(f"wrote {manifest_json}")
    print(f"stages={len(counts)} total_selected={len(selected_records)}")
    print(
        "thresholds="
        + ",".join(f"{stage['threshold']:.9f}" for stage in manifest["stages"])
    )


if __name__ == "__main__":
    main()
