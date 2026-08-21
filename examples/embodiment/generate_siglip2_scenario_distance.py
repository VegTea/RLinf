#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from PIL import Image


DEFAULT_SCENARIO_FILE = (
    "rlinf/assets_isaaclab/all_setting/combined_with_distance/"
    "Isaaclab_all_scenarios_10025_with_distance.jsonl"
)
DEFAULT_IMAGE_DIR = (
    "logs/20260629-17:54:39-全1wsetting初始图像/scenario_initial_renders"
)
DEFAULT_OUTPUT_DIR = "rlinf/assets_isaaclab/all_setting/combined_with_siglip2_distance"
DEFAULT_STAGE_OUTPUT_DIR = (
    "rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_siglip2"
)
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_COUNTS = "50,100,200,400,600,800,989,1200,1500,1997,2500,3000,4048,5000,5964,7000,8014,9000,10025"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fp.write("\n")


def scenario_id(record: dict) -> str:
    return str(record["id"]).zfill(6)


def build_recursive_image_index(image_dir: Path) -> dict[str, Path]:
    index = {}
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        candidates = [path.stem]
        if path.stem.startswith("scenario_"):
            parts = path.stem.split("_")
            if len(parts) >= 2:
                candidates.append(parts[1])
        for candidate in candidates:
            if candidate.isdigit() and candidate.zfill(6) not in index:
                index[candidate.zfill(6)] = path
    return index


def resolve_image_paths(
    records: list[dict],
    image_dir: Path,
    image_pattern: str,
    recursive_index: bool,
) -> list[Path]:
    index = build_recursive_image_index(image_dir) if recursive_index else None
    image_paths = []
    missing = []
    for record in records:
        sid = scenario_id(record)
        path = None
        if index is not None:
            path = index.get(sid)
        if path is None:
            candidate = image_dir / image_pattern.format(id=sid, raw_id=str(record["id"]))
            if candidate.exists():
                path = candidate
        if path is None:
            missing.append(sid)
        else:
            image_paths.append(path)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} rendered images under {image_dir}. "
            f"First missing ids: {missing[:20]}"
        )
    return image_paths


def load_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with path.open("rb") as fp:
            image = Image.open(fp).convert("RGB")
            images.append(image.copy())
    return images


def model_tag(model_name_or_path: str) -> str:
    return Path(model_name_or_path.rstrip("/")).name.replace("/", "-")


def build_stage_manifest(
    source_file: Path,
    selected_records: list[dict],
    counts: list[int],
    distance_key: str,
    scenario_file: Path,
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
                "incremental_ids": [str(record["id"]).zfill(6) for record in incremental],
                "cumulative_ids": [str(record["id"]).zfill(6) for record in cumulative],
            }
        )
    return {
        "source_file": str(source_file),
        "distance_key": distance_key,
        "selection": "lowest_distance",
        "total_selected": len(selected_records),
        "cumulative_counts": counts,
        "stages": stages,
        "scenario_file": str(scenario_file),
    }


def write_stage_outputs(
    output_dir: Path,
    prefix: str,
    source_file: Path,
    records_with_distance: list[dict],
    counts: list[int],
    distance_key: str,
) -> tuple[Path, Path]:
    if counts[-1] > len(records_with_distance):
        raise ValueError(
            f"largest count {counts[-1]} exceeds available records {len(records_with_distance)}"
        )
    sorted_records = sorted(
        records_with_distance,
        key=lambda record: (float(record[distance_key]), str(record["id"]).zfill(6)),
    )
    selected_records = sorted_records[: counts[-1]]
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_jsonl = output_dir / f"{prefix}_with_distance.jsonl"
    manifest_json = output_dir / f"{prefix}_stage_manifest.json"
    write_jsonl(selected_jsonl, selected_records)

    manifest = build_stage_manifest(
        source_file=source_file,
        selected_records=selected_records,
        counts=counts,
        distance_key=distance_key,
        scenario_file=selected_jsonl,
    )
    with manifest_json.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    stages_dir = output_dir / "stage_sets"
    incremental_dir = stages_dir / "incremental"
    cumulative_dir = stages_dir / "cumulative"
    records_by_id = {str(record["id"]).zfill(6): record for record in selected_records}
    for stage in manifest["stages"]:
        stage_index = int(stage["stage_index"])
        write_jsonl(
            incremental_dir / f"stage_{stage_index:02d}.jsonl",
            [records_by_id[sid] for sid in stage["incremental_ids"]],
        )
        write_jsonl(
            cumulative_dir / f"stage_{stage_index:02d}.jsonl",
            [records_by_id[sid] for sid in stage["cumulative_ids"]],
        )
    return selected_jsonl, manifest_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SigLIP2 RGB embeddings and reference cosine distances."
    )
    parser.add_argument("--scenario-file", type=Path, default=Path(DEFAULT_SCENARIO_FILE))
    parser.add_argument("--image-dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument("--image-pattern", default="scenario_{id}_table.png")
    parser.add_argument("--recursive-index", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage-output-dir", type=Path, default=Path(DEFAULT_STAGE_OUTPUT_DIR))
    parser.add_argument("--stage-prefix", default="Isaaclab_19stage_10025_siglip2")
    parser.add_argument("--counts", default=DEFAULT_COUNTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the HuggingFace model/processor from the local cache only.",
    )
    parser.add_argument("--reference-id", default="009876")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-records", type=int, default=None)
    parser.add_argument("--no-stage-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.scenario_file)
    if args.num_records is not None:
        records = records[: args.num_records]
    ids = [scenario_id(record) for record in records]
    ref_id = args.reference_id.zfill(6)
    if ref_id not in ids:
        raise ValueError(f"reference id {ref_id!r} not found in selected records")
    image_paths = resolve_image_paths(
        records=records,
        image_dir=args.image_dir,
        image_pattern=args.image_pattern,
        recursive_index=args.recursive_index,
    )

    print(f"records={len(records)}")
    print(f"image_dir={args.image_dir}")
    print(f"model={args.model}")
    print(f"reference_id={ref_id}")
    print(f"batch_size={args.batch_size}")
    print(f"first_image={image_paths[0]}")
    if args.dry_run:
        print("dry-run OK")
        return

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.eval().to(device)

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), args.batch_size):
            end = min(start + args.batch_size, len(image_paths))
            images = load_images(image_paths[start:end])
            inputs = processor(images=images, return_tensors="pt").to(device)
            output = model.get_image_features(**inputs)
            if not torch.is_tensor(output):
                if hasattr(output, "image_embeds") and output.image_embeds is not None:
                    output = output.image_embeds
                elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                    output = output.pooler_output
                elif hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state.mean(dim=1)
                else:
                    raise TypeError(
                        "Unsupported get_image_features output type: "
                        f"{type(output).__name__}"
                    )
            output = F.normalize(output.float(), dim=1)
            embeddings.append(output.cpu().numpy().astype(np.float32))
            print(f"embedded {end}/{len(image_paths)}", flush=True)

    embedding_matrix = np.concatenate(embeddings, axis=0)
    ref_index = ids.index(ref_id)
    ref_embedding = embedding_matrix[ref_index]
    similarities = embedding_matrix @ ref_embedding
    distances = 1.0 - similarities
    distances[ref_index] = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = model_tag(args.model)
    distance_key = "siglip2_cosine_distance"
    embeddings_path = args.output_dir / "siglip2_embeddings_l2norm.npy"
    metadata_path = args.output_dir / "siglip2_metadata.csv"
    output_jsonl = args.output_dir / "Isaaclab_all_scenarios_10025_with_siglip2_distance.jsonl"
    summary_path = args.output_dir / "siglip2_distance_summary.json"

    np.save(embeddings_path, embedding_matrix)
    with metadata_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "row_index",
                "scenario_id",
                "image_path",
                distance_key,
            ],
        )
        writer.writeheader()
        for idx, (sid, image_path, distance) in enumerate(zip(ids, image_paths, distances)):
            writer.writerow(
                {
                    "row_index": idx,
                    "scenario_id": sid,
                    "image_path": str(image_path),
                    distance_key: float(distance),
                }
            )

    output_records = []
    for record, distance in zip(records, distances):
        output_record = dict(record)
        output_record[distance_key] = float(distance)
        output_record["siglip2_reference_id"] = ref_id
        output_record["siglip2_model"] = args.model
        output_records.append(output_record)
    write_jsonl(output_jsonl, output_records)

    sorted_pairs = sorted(zip(distances.tolist(), ids), key=lambda item: (item[0], item[1]))
    summary = {
        "scenario_file": str(args.scenario_file),
        "image_dir": str(args.image_dir),
        "image_pattern": args.image_pattern,
        "recursive_index": bool(args.recursive_index),
        "model": args.model,
        "model_tag": tag,
        "reference_id": ref_id,
        "count": len(ids),
        "embedding_dim": int(embedding_matrix.shape[1]),
        "distance_key": distance_key,
        "distance_min": float(distances.min()),
        "distance_max": float(distances.max()),
        "distance_mean": float(distances.mean()),
        "first_20_ids": [sid for _, sid in sorted_pairs[:20]],
        "last_20_ids": [sid for _, sid in sorted_pairs[-20:]],
        "embeddings": str(embeddings_path),
        "metadata": str(metadata_path),
        "output_jsonl": str(output_jsonl),
    }

    if not args.no_stage_manifest:
        counts = parse_counts(args.counts)
        selected_jsonl, manifest_json = write_stage_outputs(
            output_dir=args.stage_output_dir,
            prefix=args.stage_prefix,
            source_file=output_jsonl,
            records_with_distance=output_records,
            counts=counts,
            distance_key=distance_key,
        )
        summary["stage_output_dir"] = str(args.stage_output_dir)
        summary["stage_scenario_file"] = str(selected_jsonl)
        summary["stage_manifest"] = str(manifest_json)
        summary["stage_counts"] = counts

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    print(f"wrote {embeddings_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {output_jsonl}")
    print(f"wrote {summary_path}")
    if not args.no_stage_manifest:
        print(f"wrote {summary['stage_scenario_file']}")
        print(f"wrote {summary['stage_manifest']}")
    print(
        f"{distance_key}="
        f"{summary['distance_min']:.9f}..{summary['distance_max']:.9f}, "
        f"mean={summary['distance_mean']:.9f}"
    )


if __name__ == "__main__":
    main()
