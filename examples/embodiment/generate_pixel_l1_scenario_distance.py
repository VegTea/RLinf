#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SCENARIO_FILE = (
    "rlinf/assets_isaaclab/all_setting/combined_with_distance/"
    "Isaaclab_all_scenarios_10025_with_distance.jsonl"
)
DEFAULT_IMAGE_DIR = (
    "logs/20260629-17:54:39-全1wsetting初始图像/scenario_initial_renders"
)
DEFAULT_OUTPUT_DIR = "rlinf/assets_isaaclab/all_setting/combined_with_pixel_l1_distance"


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
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


def scenario_id(record: dict) -> str:
    return str(record["id"]).zfill(6)


def image_path(image_dir: Path, image_pattern: str, record: dict) -> Path:
    sid = scenario_id(record)
    return image_dir / image_pattern.format(id=sid, raw_id=str(record["id"]))


def load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with path.open("rb") as fp:
        image = Image.open(fp).convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute model-free RGB pixel L1 distances to one reference image."
    )
    parser.add_argument("--scenario-file", type=Path, default=Path(DEFAULT_SCENARIO_FILE))
    parser.add_argument("--image-dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument("--image-pattern", default="scenario_{id}_table.png")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reference-id", default="009876")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--num-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.scenario_file)
    if args.num_records is not None:
        records = records[: args.num_records]
    ids = [scenario_id(record) for record in records]
    ref_id = args.reference_id.zfill(6)
    if ref_id not in ids:
        raise ValueError(f"reference id {ref_id!r} not found in selected records")

    missing = []
    paths = []
    for record in records:
        path = image_path(args.image_dir, args.image_pattern, record)
        if not path.exists():
            missing.append(str(path))
        paths.append(path)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} rendered images. First missing: {missing[:10]}"
        )

    print(f"records={len(records)}")
    print(f"image_dir={args.image_dir}")
    print(f"reference_id={ref_id}")
    print(f"resize={args.resize}")
    print(f"first_image={paths[0]}")
    if args.dry_run:
        print("dry-run OK")
        return

    size = (args.resize, args.resize)
    ref_record = records[ids.index(ref_id)]
    ref_image = load_rgb(image_path(args.image_dir, args.image_pattern, ref_record), size)

    distances = []
    for index, path in enumerate(paths, 1):
        image = load_rgb(path, size)
        distances.append(float(np.mean(np.abs(image - ref_image))))
        if index % 1000 == 0 or index == len(paths):
            print(f"computed {index}/{len(paths)}", flush=True)

    output_records = []
    for record, distance in zip(records, distances):
        output = dict(record)
        output["pixel_l1_distance"] = distance
        output["pixel_l1_reference_id"] = ref_id
        output["pixel_l1_resize"] = args.resize
        output_records.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "Isaaclab_all_scenarios_10025_with_pixel_l1_distance.jsonl"
    summary_path = args.output_dir / "pixel_l1_distance_summary.json"
    write_jsonl(output_jsonl, output_records)

    sorted_pairs = sorted(zip(distances, ids), key=lambda item: (item[0], item[1]))
    summary = {
        "scenario_file": str(args.scenario_file),
        "image_dir": str(args.image_dir),
        "image_pattern": args.image_pattern,
        "reference_id": ref_id,
        "count": len(records),
        "resize": args.resize,
        "distance_key": "pixel_l1_distance",
        "distance_min": float(min(distances)),
        "distance_max": float(max(distances)),
        "distance_mean": float(sum(distances) / len(distances)),
        "first_20_ids": [sid for _, sid in sorted_pairs[:20]],
        "last_20_ids": [sid for _, sid in sorted_pairs[-20:]],
        "output_jsonl": str(output_jsonl),
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    print(f"wrote {output_jsonl}")
    print(f"wrote {summary_path}")
    print(
        "pixel_l1_distance="
        f"{summary['distance_min']:.9f}..{summary['distance_max']:.9f}, "
        f"mean={summary['distance_mean']:.9f}"
    )


if __name__ == "__main__":
    main()
