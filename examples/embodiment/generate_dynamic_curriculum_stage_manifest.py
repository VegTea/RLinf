#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_COUNTS_10STAGE_200 = "10,20,40,60,80,100,120,140,170,200"
DEFAULT_COUNTS_19STAGE_10025 = (
    "50,100,200,400,600,800,989,1200,1500,1997,2500,3000,"
    "4048,5000,5964,7000,8014,9000,10025"
)


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


def load_metadata_ids(metadata_csv: Path) -> list[str]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        ids = [str(row["scenario_id"]) for row in reader]
    if not ids:
        raise ValueError(f"No scenario ids found in {metadata_csv}")
    return ids


def topk_mean_distance_to_mastered(
    embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    mastered_indices: np.ndarray,
    top_k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_eff = min(int(top_k), int(mastered_indices.shape[0]))
    if k_eff <= 0:
        raise ValueError("mastered set must not be empty")

    dynamic_distances = np.empty(candidate_indices.shape[0], dtype=np.float32)
    nearest_mastered_positions = np.empty((candidate_indices.shape[0], k_eff), dtype=np.int64)
    nearest_distances = np.empty((candidate_indices.shape[0], k_eff), dtype=np.float32)
    mastered_embeddings = embeddings[mastered_indices]

    for start in range(0, candidate_indices.shape[0], chunk_size):
        end = min(start + chunk_size, candidate_indices.shape[0])
        candidate_embeddings = embeddings[candidate_indices[start:end]]
        similarity = candidate_embeddings @ mastered_embeddings.T
        distances = 1.0 - similarity
        if k_eff == distances.shape[1]:
            top_positions = np.argsort(distances, axis=1)[:, :k_eff]
        else:
            top_positions = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
            top_values_unsorted = np.take_along_axis(distances, top_positions, axis=1)
            order = np.argsort(top_values_unsorted, axis=1)
            top_positions = np.take_along_axis(top_positions, order, axis=1)
        top_values = np.take_along_axis(distances, top_positions, axis=1)
        dynamic_distances[start:end] = top_values.mean(axis=1)
        nearest_mastered_positions[start:end] = top_positions
        nearest_distances[start:end] = top_values

    return dynamic_distances, nearest_mastered_positions, nearest_distances


def build_dynamic_curriculum(
    records: list[dict],
    ids: list[str],
    embeddings: np.ndarray,
    counts: list[int],
    reference_id: str,
    top_k: int,
    chunk_size: int,
) -> tuple[list[dict], dict]:
    id_to_record = {str(record["id"]): record for record in records}
    missing_records = [scenario_id for scenario_id in ids if scenario_id not in id_to_record]
    if missing_records:
        preview = ", ".join(missing_records[:10])
        raise ValueError(f"Embedding metadata ids missing from scenario jsonl: {preview}")

    if counts[-1] > len(ids):
        raise ValueError(f"largest count {counts[-1]} exceeds available ids {len(ids)}")
    if reference_id not in ids:
        raise ValueError(f"reference_id {reference_id!r} not found in metadata ids")

    id_to_index = {scenario_id: idx for idx, scenario_id in enumerate(ids)}
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    stages = []
    dynamic_info_by_id: dict[str, dict] = {}

    all_indices = np.arange(len(ids), dtype=np.int64)
    reference_idx = id_to_index[reference_id]
    reference_distances = 1.0 - (embeddings @ embeddings[reference_idx])
    reference_distances[reference_idx] = 0.0

    for stage_index, cumulative_count in enumerate(counts):
        previous_count = len(selected_indices)
        incremental_count = cumulative_count - previous_count
        if incremental_count <= 0:
            raise ValueError(
                f"Stage {stage_index} has non-positive incremental count: "
                f"{incremental_count}"
            )

        candidate_indices = np.array(
            [idx for idx in all_indices.tolist() if idx not in selected_set],
            dtype=np.int64,
        )
        if stage_index == 0:
            candidate_scores = reference_distances[candidate_indices]
            order = np.lexsort(
                (
                    np.array([ids[idx] for idx in candidate_indices]),
                    candidate_scores,
                )
            )
            chosen = candidate_indices[order[:incremental_count]]
            nearest_positions = np.zeros((chosen.shape[0], 1), dtype=np.int64)
            nearest_dists = reference_distances[chosen].reshape(-1, 1)
            stage_distance_key = "reference_cosine_distance"
        else:
            mastered_indices = np.array(selected_indices, dtype=np.int64)
            scores, nearest_positions, nearest_dists = topk_mean_distance_to_mastered(
                embeddings=embeddings,
                candidate_indices=candidate_indices,
                mastered_indices=mastered_indices,
                top_k=top_k,
                chunk_size=chunk_size,
            )
            order = np.lexsort(
                (
                    np.array([ids[idx] for idx in candidate_indices]),
                    scores,
                )
            )
            selected_rows = order[:incremental_count]
            chosen = candidate_indices[selected_rows]
            nearest_positions = nearest_positions[selected_rows]
            nearest_dists = nearest_dists[selected_rows]
            stage_distance_key = f"dynamic_top{top_k}_mean_cosine_distance"

        incremental_ids = [ids[int(idx)] for idx in chosen]
        for local_idx, scenario_idx in enumerate(chosen.tolist()):
            scenario_id = ids[scenario_idx]
            if stage_index == 0:
                nearest_ids = [reference_id]
            else:
                mastered_at_selection = selected_indices
                nearest_ids = [
                    ids[mastered_at_selection[int(pos)]]
                    for pos in nearest_positions[local_idx].tolist()
                ]
            dynamic_info_by_id[scenario_id] = {
                "selected_stage": stage_index,
                stage_distance_key: float(nearest_dists[local_idx].mean()),
                "nearest_mastered_ids": nearest_ids,
                "nearest_mastered_distances": [
                    float(value) for value in nearest_dists[local_idx].tolist()
                ],
            }

        selected_indices.extend([int(idx) for idx in chosen.tolist()])
        selected_set.update(int(idx) for idx in chosen.tolist())

        cumulative_ids = [ids[idx] for idx in selected_indices]
        incremental_distances = [
            dynamic_info_by_id[scenario_id][stage_distance_key]
            for scenario_id in incremental_ids
        ]
        cumulative_dynamic_distances = []
        for scenario_id in cumulative_ids:
            info = dynamic_info_by_id[scenario_id]
            if "dynamic_top10_mean_cosine_distance" in info:
                cumulative_dynamic_distances.append(
                    info["dynamic_top10_mean_cosine_distance"]
                )
            elif "reference_cosine_distance" in info:
                cumulative_dynamic_distances.append(info["reference_cosine_distance"])

        stages.append(
            {
                "stage_index": stage_index,
                "cumulative_count": len(cumulative_ids),
                "incremental_count": len(incremental_ids),
                "distance_key": stage_distance_key,
                "threshold": float(max(incremental_distances)),
                "incremental_min_distance": float(min(incremental_distances)),
                "incremental_max_distance": float(max(incremental_distances)),
                "cumulative_min_distance": float(min(cumulative_dynamic_distances)),
                "cumulative_max_distance": float(max(cumulative_dynamic_distances)),
                "incremental_ids": incremental_ids,
                "cumulative_ids": cumulative_ids,
            }
        )

    selected_records = []
    for scenario_id in [ids[idx] for idx in selected_indices]:
        record = dict(id_to_record[scenario_id])
        record.update(dynamic_info_by_id[scenario_id])
        selected_records.append(record)

    manifest = {
        "selection": "dynamic_topk_region_growing",
        "reference_id": reference_id,
        "top_k": top_k,
        "total_selected": len(selected_records),
        "cumulative_counts": counts,
        "stages": stages,
    }
    return selected_records, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dynamic top-k curriculum stage manifests."
    )
    parser.add_argument("--scenario-file", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reference-id", default="009876")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    counts = parse_counts(args.counts)
    records = read_jsonl(args.scenario_file)
    ids = load_metadata_ids(args.metadata_csv)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    if embeddings.shape[0] != len(ids):
        raise ValueError(
            f"embedding rows {embeddings.shape[0]} != metadata rows {len(ids)}"
        )

    selected_records, manifest = build_dynamic_curriculum(
        records=records,
        ids=ids,
        embeddings=embeddings,
        counts=counts,
        reference_id=args.reference_id,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_jsonl = args.output_dir / f"{args.prefix}_with_dynamic_distance.jsonl"
    manifest_json = args.output_dir / f"{args.prefix}_stage_manifest.json"
    write_jsonl(selected_jsonl, selected_records)
    manifest["scenario_file"] = str(selected_jsonl)
    manifest["source_scenario_file"] = str(args.scenario_file)
    manifest["metadata_csv"] = str(args.metadata_csv)
    manifest["embeddings"] = str(args.embeddings)
    with manifest_json.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    records_by_id = {str(record["id"]): record for record in selected_records}
    for kind in ("incremental", "cumulative"):
        (args.output_dir / "stage_sets" / kind).mkdir(parents=True, exist_ok=True)
    for stage in manifest["stages"]:
        stage_index = int(stage["stage_index"])
        write_jsonl(
            args.output_dir / "stage_sets" / "incremental" / f"stage_{stage_index:02d}.jsonl",
            [records_by_id[scenario_id] for scenario_id in stage["incremental_ids"]],
        )
        write_jsonl(
            args.output_dir / "stage_sets" / "cumulative" / f"stage_{stage_index:02d}.jsonl",
            [records_by_id[scenario_id] for scenario_id in stage["cumulative_ids"]],
        )

    print(f"wrote {selected_jsonl}")
    print(f"wrote {manifest_json}")
    print(f"stages={len(counts)} total_selected={len(selected_records)} top_k={args.top_k}")
    for stage in manifest["stages"]:
        print(
            f"stage={stage['stage_index']:02d} cumulative={stage['cumulative_count']} "
            f"incremental={stage['incremental_count']} threshold={stage['threshold']:.9f}"
        )


if __name__ == "__main__":
    main()
