#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


DEFAULT_SCENARIO_FILE = (
    "rlinf/assets_isaaclab/all_setting/combined_with_distance/"
    "Isaaclab_all_scenarios_10025_with_distance.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "rlinf/assets_isaaclab/all_setting/combined_with_imagebind_distance"
)
DEFAULT_IMAGEBIND_REPO = (
    "/inspire/hdd/global_user/gongjingjing-25039/xpyu/project/ImageBind"
)
DEFAULT_IMAGE_DIR = (
    "logs/20260629-17:54:39-全1wsetting初始图像/scenario_initial_renders"
)


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


def build_recursive_image_index(image_dir: Path) -> dict[str, Path]:
    index = {}
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stem = path.stem
        candidates = []
        if stem.startswith("scenario_"):
            parts = stem.split("_")
            if len(parts) >= 2:
                candidates.append(parts[1])
        candidates.append(stem)
        for candidate in candidates:
            if candidate.isdigit() and candidate not in index:
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


def load_image_batch(paths: list[Path], transform, device: torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        with path.open("rb") as fp:
            image = Image.open(fp).convert("RGB")
        images.append(transform(image))
    return torch.stack(images, dim=0).to(device, non_blocking=True)


def load_imagebind_model(imagebind_repo: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(imagebind_repo))
    from imagebind.models import imagebind_model

    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ImageBind RGB embeddings and reference cosine distances."
    )
    parser.add_argument("--scenario-file", type=Path, default=Path(DEFAULT_SCENARIO_FILE))
    parser.add_argument("--image-dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument(
        "--image-pattern",
        default="scenario_{id}_table.png",
        help="Relative filename pattern under --image-dir. Supports {id} and {raw_id}.",
    )
    parser.add_argument(
        "--recursive-index",
        action="store_true",
        help="Index images recursively under --image-dir instead of using --image-pattern only.",
    )
    parser.add_argument("--imagebind-repo", type=Path, default=Path(DEFAULT_IMAGEBIND_REPO))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(DEFAULT_IMAGEBIND_REPO) / ".checkpoints" / "imagebind_huge.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reference-id", default="009876")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.scenario_file)
    if args.num_records is not None:
        records = records[: args.num_records]
    ids = [scenario_id(record) for record in records]
    if not args.dry_run and args.reference_id.zfill(6) not in ids:
        raise ValueError(f"reference id {args.reference_id!r} not found in selected records")
    if not args.imagebind_repo.exists():
        raise FileNotFoundError(f"ImageBind repo not found: {args.imagebind_repo}")

    image_paths = resolve_image_paths(
        records=records,
        image_dir=args.image_dir,
        image_pattern=args.image_pattern,
        recursive_index=args.recursive_index,
    )
    print(f"records={len(records)}")
    print(f"image_dir={args.image_dir}")
    print(f"checkpoint={args.checkpoint}")
    print(f"first_image={image_paths[0]}")
    if args.dry_run:
        print("dry-run OK")
        return
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"ImageBind checkpoint not found: {args.checkpoint}. "
            "Run examples/embodiment/setup_imagebind_env.sh with DOWNLOAD_CHECKPOINT=1 "
            "or pass --checkpoint."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    model = load_imagebind_model(args.imagebind_repo, args.checkpoint, device)
    from imagebind.models.imagebind_model import ModalityType

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), args.batch_size):
            end = min(start + args.batch_size, len(image_paths))
            batch = load_image_batch(image_paths[start:end], transform, device)
            output = model({ModalityType.VISION: batch})[ModalityType.VISION]
            output = F.normalize(output.float(), dim=1)
            embeddings.append(output.cpu().numpy().astype(np.float32))
            print(f"embedded {end}/{len(image_paths)}", flush=True)

    embedding_matrix = np.concatenate(embeddings, axis=0)
    ref_index = ids.index(args.reference_id.zfill(6))
    ref_embedding = embedding_matrix[ref_index]
    similarities = embedding_matrix @ ref_embedding
    distances = 1.0 - similarities
    distances[ref_index] = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / "imagebind_embeddings_l2norm.npy"
    metadata_path = args.output_dir / "imagebind_metadata.csv"
    output_jsonl = args.output_dir / "Isaaclab_all_scenarios_10025_with_imagebind_distance.jsonl"
    summary_path = args.output_dir / "imagebind_distance_summary.json"

    np.save(embeddings_path, embedding_matrix)
    with metadata_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "row_index",
                "scenario_id",
                "image_path",
                "imagebind_cosine_distance",
            ],
        )
        writer.writeheader()
        for idx, (sid, image_path, distance) in enumerate(zip(ids, image_paths, distances)):
            writer.writerow(
                {
                    "row_index": idx,
                    "scenario_id": sid,
                    "image_path": str(image_path),
                    "imagebind_cosine_distance": float(distance),
                }
            )

    output_records = []
    for record, distance in zip(records, distances):
        output_record = dict(record)
        output_record["imagebind_cosine_distance"] = float(distance)
        output_record["imagebind_reference_id"] = args.reference_id.zfill(6)
        output_record["imagebind_model"] = "imagebind_huge"
        output_records.append(output_record)
    write_jsonl(output_jsonl, output_records)

    sorted_pairs = sorted(zip(distances.tolist(), ids), key=lambda item: (item[0], item[1]))
    summary = {
        "scenario_file": str(args.scenario_file),
        "image_dir": str(args.image_dir),
        "image_pattern": args.image_pattern,
        "recursive_index": bool(args.recursive_index),
        "checkpoint": str(args.checkpoint),
        "reference_id": args.reference_id.zfill(6),
        "count": len(ids),
        "embedding_dim": int(embedding_matrix.shape[1]),
        "distance_key": "imagebind_cosine_distance",
        "distance_min": float(distances.min()),
        "distance_max": float(distances.max()),
        "distance_mean": float(distances.mean()),
        "first_20_ids": [sid for _, sid in sorted_pairs[:20]],
        "last_20_ids": [sid for _, sid in sorted_pairs[-20:]],
        "embeddings": str(embeddings_path),
        "metadata": str(metadata_path),
        "output_jsonl": str(output_jsonl),
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    print(f"wrote {embeddings_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {output_jsonl}")
    print(f"wrote {summary_path}")
    print(
        "imagebind_cosine_distance="
        f"{summary['distance_min']:.9f}..{summary['distance_max']:.9f}, "
        f"mean={summary['distance_mean']:.9f}"
    )


if __name__ == "__main__":
    main()
