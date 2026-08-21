#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fp.write("\n")


def load_candidate_ids(metadata_csv: Path) -> list[str]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        ids = [str(row["scenario_id"]) for row in reader]
    if not ids:
        raise ValueError(f"No scenario ids found in {metadata_csv}")
    return ids


def topk_mean_distance(
    candidate_embeddings: np.ndarray,
    mastered_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_eff = min(int(top_k), int(mastered_embeddings.shape[0]))
    if k_eff <= 0:
        raise ValueError("mastered embeddings must not be empty")
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
    return top_values.mean(axis=1).astype(np.float32), top_positions, top_values.astype(np.float32)


def score_candidates_to_mastered(
    candidate_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    mastered_embeddings: np.ndarray,
    top_k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_eff = min(int(top_k), int(mastered_embeddings.shape[0]))
    scores = np.empty(candidate_indices.shape[0], dtype=np.float32)
    nearest_positions = np.empty((candidate_indices.shape[0], k_eff), dtype=np.int64)
    nearest_distances = np.empty((candidate_indices.shape[0], k_eff), dtype=np.float32)

    for start in range(0, candidate_indices.shape[0], chunk_size):
        end = min(start + chunk_size, candidate_indices.shape[0])
        chunk_embeddings = candidate_embeddings[candidate_indices[start:end]]
        chunk_scores, chunk_positions, chunk_distances = topk_mean_distance(
            chunk_embeddings, mastered_embeddings, top_k
        )
        scores[start:end] = chunk_scores
        nearest_positions[start:end] = chunk_positions
        nearest_distances[start:end] = chunk_distances

    return scores, nearest_positions, nearest_distances


def build_curriculum_from_base(
    records: list[dict],
    candidate_ids: list[str],
    candidate_embeddings: np.ndarray,
    base_embeddings: np.ndarray,
    counts: list[int],
    top_k: int,
    chunk_size: int,
) -> tuple[list[dict], dict, dict]:
    id_to_record = {str(record["id"]): record for record in records}
    missing_records = [scenario_id for scenario_id in candidate_ids if scenario_id not in id_to_record]
    if missing_records:
        raise ValueError(
            "candidate metadata ids missing from scenario jsonl: "
            + ", ".join(missing_records[:10])
        )
    if counts[-1] > len(candidate_ids):
        raise ValueError(
            f"largest count {counts[-1]} exceeds available candidates {len(candidate_ids)}"
        )
    if candidate_embeddings.shape[0] != len(candidate_ids):
        raise ValueError(
            f"candidate embedding rows {candidate_embeddings.shape[0]} != ids {len(candidate_ids)}"
        )
    if candidate_embeddings.shape[1] != base_embeddings.shape[1]:
        raise ValueError(
            "candidate/base embedding dims mismatch: "
            f"{candidate_embeddings.shape[1]} != {base_embeddings.shape[1]}"
        )

    all_indices = np.arange(len(candidate_ids), dtype=np.int64)
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    stages = []
    dynamic_info_by_id: dict[str, dict] = {}

    for stage_index, cumulative_count in enumerate(counts):
        previous_count = len(selected_indices)
        incremental_count = cumulative_count - previous_count
        if incremental_count <= 0:
            raise ValueError(
                f"Stage {stage_index} has non-positive incremental count: {incremental_count}"
            )

        candidate_indices = np.array(
            [idx for idx in all_indices.tolist() if idx not in selected_set],
            dtype=np.int64,
        )
        if selected_indices:
            mastered_embeddings = np.concatenate(
                [base_embeddings, candidate_embeddings[np.array(selected_indices, dtype=np.int64)]],
                axis=0,
            )
        else:
            mastered_embeddings = base_embeddings

        scores, nearest_positions, nearest_distances = score_candidates_to_mastered(
            candidate_embeddings=candidate_embeddings,
            candidate_indices=candidate_indices,
            mastered_embeddings=mastered_embeddings,
            top_k=top_k,
            chunk_size=chunk_size,
        )
        order = np.lexsort(
            (
                np.array([candidate_ids[idx] for idx in candidate_indices]),
                scores,
            )
        )
        selected_rows = order[:incremental_count]
        chosen = candidate_indices[selected_rows]
        chosen_scores = scores[selected_rows]
        chosen_nearest_positions = nearest_positions[selected_rows]
        chosen_nearest_distances = nearest_distances[selected_rows]

        incremental_ids = [candidate_ids[int(idx)] for idx in chosen]
        base_count = int(base_embeddings.shape[0])
        for local_idx, scenario_idx in enumerate(chosen.tolist()):
            scenario_id = candidate_ids[scenario_idx]
            nearest_refs = []
            for pos in chosen_nearest_positions[local_idx].tolist():
                pos = int(pos)
                if pos < base_count:
                    nearest_refs.append(f"base:{pos}")
                else:
                    selected_pos = pos - base_count
                    nearest_refs.append(candidate_ids[selected_indices[selected_pos]])
            dynamic_info_by_id[scenario_id] = {
                "selected_stage": stage_index,
                f"base_dynamic_top{top_k}_mean_cosine_distance": float(chosen_scores[local_idx]),
                "nearest_mastered_refs": nearest_refs,
                "nearest_mastered_distances": [
                    float(value) for value in chosen_nearest_distances[local_idx].tolist()
                ],
            }

        selected_indices.extend([int(idx) for idx in chosen.tolist()])
        selected_set.update(int(idx) for idx in chosen.tolist())

        cumulative_ids = [candidate_ids[idx] for idx in selected_indices]
        incremental_distances = [
            dynamic_info_by_id[scenario_id][f"base_dynamic_top{top_k}_mean_cosine_distance"]
            for scenario_id in incremental_ids
        ]
        cumulative_distances = [
            dynamic_info_by_id[scenario_id][f"base_dynamic_top{top_k}_mean_cosine_distance"]
            for scenario_id in cumulative_ids
        ]

        previous_threshold = None if stage_index == 0 else stages[-1]["threshold"]
        threshold = float(max(cumulative_distances))
        stages.append(
            {
                "stage_index": stage_index,
                "cumulative_count": len(cumulative_ids),
                "incremental_count": len(incremental_ids),
                "distance_key": f"base_dynamic_top{top_k}_mean_cosine_distance",
                "threshold": threshold,
                "previous_threshold": previous_threshold,
                "range": f"dynamic top-{top_k} incremental selection stage {stage_index}",
                "incremental_min_distance": float(min(incremental_distances)),
                "incremental_max_distance": float(max(incremental_distances)),
                "cumulative_min_distance": float(min(cumulative_distances)),
                "cumulative_max_distance": float(max(cumulative_distances)),
                "incremental_ids": incremental_ids,
                "cumulative_ids": cumulative_ids,
            }
        )

    selected_records = []
    for scenario_id in [candidate_ids[idx] for idx in selected_indices]:
        record = dict(id_to_record[scenario_id])
        record.update(dynamic_info_by_id[scenario_id])
        selected_records.append(record)

    manifest = {
        "selection": "base_embedding_dynamic_topk_region_growing",
        "top_k": top_k,
        "base_embedding_count": int(base_embeddings.shape[0]),
        "total_selected": len(selected_records),
        "cumulative_counts": counts,
        "stages": stages,
    }
    eval_summary = {
        "description": "Incremental scenario ids generated from base embeddings by dynamic top-k region growing.",
        "distance_key": f"base_dynamic_top{top_k}_mean_cosine_distance",
        "total_scenarios": len(selected_records),
        "thresholds": [stage["threshold"] for stage in stages],
        "stage_count": len(stages),
        "stages": [
            {
                "stage_index": stage["stage_index"],
                "threshold": stage["threshold"],
                "previous_threshold": stage["previous_threshold"],
                "range": stage["range"],
                "new_count": stage["incremental_count"],
                "cumulative_count": stage["cumulative_count"],
                "new_min_distance": stage["incremental_min_distance"],
                "new_max_distance": stage["incremental_max_distance"],
                "scenario_ids": stage["incremental_ids"],
            }
            for stage in stages
        ],
    }
    return selected_records, manifest, eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dynamic top-k curriculum stages from a base embedding set."
    )
    parser.add_argument("--scenario-file", required=True, type=Path)
    parser.add_argument("--candidate-metadata-csv", required=True, type=Path)
    parser.add_argument("--candidate-embeddings", required=True, type=Path)
    parser.add_argument("--base-embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--counts", default=DEFAULT_COUNTS_19STAGE_10025)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    counts = parse_counts(args.counts)
    records = read_jsonl(args.scenario_file)
    candidate_ids = load_candidate_ids(args.candidate_metadata_csv)
    candidate_embeddings = np.load(args.candidate_embeddings, mmap_mode="r")
    base_embeddings = np.asarray(np.load(args.base_embeddings, mmap_mode="r"), dtype=np.float32)

    selected_records, manifest, eval_summary = build_curriculum_from_base(
        records=records,
        candidate_ids=candidate_ids,
        candidate_embeddings=candidate_embeddings,
        base_embeddings=base_embeddings,
        counts=counts,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_jsonl = args.output_dir / f"{args.prefix}_with_base_dynamic_distance.jsonl"
    manifest_json = args.output_dir / f"{args.prefix}_stage_manifest.json"
    summary_json = args.output_dir / f"{args.prefix}_eval_summary.json"

    write_jsonl(selected_jsonl, selected_records)
    manifest.update(
        {
            "scenario_file": str(selected_jsonl),
            "source_scenario_file": str(args.scenario_file),
            "candidate_metadata_csv": str(args.candidate_metadata_csv),
            "candidate_embeddings": str(args.candidate_embeddings),
            "base_embeddings": str(args.base_embeddings),
        }
    )
    eval_summary.update(
        {
            "scenario_file": str(selected_jsonl),
            "source_scenario_file": str(args.scenario_file),
            "stage_manifest_file": str(manifest_json),
            "base_embeddings": str(args.base_embeddings),
        }
    )
    manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(eval_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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
    print(f"wrote {summary_json}")
    print(
        f"stages={len(counts)} total_selected={len(selected_records)} "
        f"base_embeddings={base_embeddings.shape[0]} top_k={args.top_k}"
    )
    for stage in manifest["stages"]:
        print(
            f"stage={stage['stage_index']:02d} cumulative={stage['cumulative_count']} "
            f"incremental={stage['incremental_count']} threshold={stage['threshold']:.9f}"
        )


if __name__ == "__main__":
    main()
