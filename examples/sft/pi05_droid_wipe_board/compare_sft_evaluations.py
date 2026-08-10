#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare decoded action-chunk metrics from JAX and RLinf SFT runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_EVAL_METRICS = (
    "action_normalized_mse",
    "action_normalized_mae",
    "joint_velocity_mse",
    "joint_velocity_mae",
    "gripper_position_mse",
    "gripper_position_mae",
    "joint_position_mse",
    "joint_position_mae",
    "num_chunks",
)


def load_jax_metrics(metrics_path: Path) -> list[dict[str, float | int]]:
    """Load OpenPI JAX ``eval_metrics.json`` into checkpoint-indexed rows."""
    raw_metrics = json.loads(metrics_path.read_text())
    rows = []
    for checkpoint_step, values in sorted(
        raw_metrics.items(), key=lambda item: int(item[0])
    ):
        row: dict[str, float | int] = {"checkpoint_step": int(checkpoint_step)}
        for metric in _EVAL_METRICS:
            tag = f"eval/{metric}"
            if tag not in values:
                raise KeyError(f"{tag} is missing at JAX checkpoint {checkpoint_step}.")
            row[metric] = float(values[tag])
        if "eval/time_seconds" in values:
            row["eval_time_seconds"] = float(values["eval/time_seconds"])
        rows.append(row)
    _validate_rows(rows, source=str(metrics_path))
    return rows


def _find_event_file(run_dir: Path) -> Path:
    event_files = sorted(run_dir.glob("**/events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(
            f"Expected exactly one TensorBoard event file under {run_dir}, "
            f"found {len(event_files)}."
        )
    return event_files[0]


def load_rlinf_metrics(run_dir: Path) -> list[dict[str, float | int]]:
    """Load RLinf TensorBoard metrics and map log step N-1 to checkpoint N."""
    event_file = _find_event_file(run_dir)
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    required_tags = {f"eval/{metric}" for metric in _EVAL_METRICS}
    missing_tags = sorted(required_tags - scalar_tags)
    if missing_tags:
        raise KeyError(f"Missing RLinf eval tags in {event_file}: {missing_tags}")

    by_step: dict[int, dict[str, float | int]] = {}
    for metric in _EVAL_METRICS:
        for event in accumulator.Scalars(f"eval/{metric}"):
            row = by_step.setdefault(
                event.step, {"checkpoint_step": int(event.step) + 1}
            )
            row[metric] = float(event.value)

    if "time/evaluate" in scalar_tags:
        for event in accumulator.Scalars("time/evaluate"):
            if event.step in by_step:
                by_step[event.step]["eval_time_seconds"] = float(event.value)
    if "train/loss" in scalar_tags:
        train_loss = {
            event.step: float(event.value)
            for event in accumulator.Scalars("train/loss")
        }
        for step, row in by_step.items():
            if step in train_loss:
                row["train_loss"] = train_loss[step]

    rows = [by_step[step] for step in sorted(by_step)]
    _validate_rows(rows, source=str(event_file))
    return rows


def _validate_rows(rows: list[dict[str, float | int]], source: str) -> None:
    if not rows:
        raise ValueError(f"No decoded evaluation rows found in {source}.")
    for row in rows:
        missing = [metric for metric in _EVAL_METRICS if metric not in row]
        if missing:
            raise KeyError(
                f"Evaluation row {row.get('checkpoint_step')} in {source} "
                f"is incomplete: {missing}"
            )
        numeric_values = [float(row[metric]) for metric in _EVAL_METRICS]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(
                f"Evaluation row {row.get('checkpoint_step')} in {source} "
                "contains non-finite values."
            )


def _best_row(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    return min(rows, key=lambda row: float(row["joint_position_mse"]))


def _reduction_percent(reference: float, candidate: float) -> float:
    if reference == 0:
        raise ValueError("Cannot compute a percentage reduction from zero.")
    return (reference - candidate) / reference * 100.0


def validate_expected_evaluations(
    rows: list[dict[str, float | int]],
    *,
    expected_eval_count: int,
    expected_num_chunks: int,
    source: str,
) -> None:
    """Reject partial runs or evaluations with the wrong chunk count."""
    if len(rows) != expected_eval_count:
        raise ValueError(
            f"{source} has {len(rows)} evaluation rows; expected {expected_eval_count}."
        )
    invalid_steps = [
        int(row["checkpoint_step"])
        for row in rows
        if int(row["num_chunks"]) != expected_num_chunks
    ]
    if invalid_steps:
        raise ValueError(
            f"{source} does not contain {expected_num_chunks} chunks at "
            f"checkpoints {invalid_steps}."
        )


def build_comparison(
    jax_metrics_path: Path,
    aligned_rlinf_run: Path,
    wrapper_baseline_run: Path | None = None,
    *,
    expected_eval_count: int | None = None,
    expected_num_chunks: int | None = None,
) -> dict[str, Any]:
    """Build a comparison while keeping aligned and wrapper runs distinct."""
    runs = {
        "openpi_jax": {
            "implementation": "official OpenPI JAX",
            "source": str(jax_metrics_path.resolve()),
            "evaluations": load_jax_metrics(jax_metrics_path),
        },
        "rlinf_openpi_pytorch_aligned": {
            "implementation": "RLinf openpi_pytorch (documented as openpi_rlinf)",
            "source": str(aligned_rlinf_run.resolve()),
            "evaluations": load_rlinf_metrics(aligned_rlinf_run),
        },
    }
    if wrapper_baseline_run is not None:
        runs["official_openpi_pytorch_wrapper_baseline"] = {
            "implementation": "RLinf official OpenPI PyTorch wrapper",
            "source": str(wrapper_baseline_run.resolve()),
            "evaluations": load_rlinf_metrics(wrapper_baseline_run),
        }

    if (expected_eval_count is None) != (expected_num_chunks is None):
        raise ValueError(
            "expected_eval_count and expected_num_chunks must be provided together."
        )
    if expected_eval_count is not None and expected_num_chunks is not None:
        for name, run in runs.items():
            validate_expected_evaluations(
                run["evaluations"],
                expected_eval_count=expected_eval_count,
                expected_num_chunks=expected_num_chunks,
                source=name,
            )

    best = {name: _best_row(run["evaluations"]) for name, run in runs.items()}
    final = {name: run["evaluations"][-1] for name, run in runs.items()}
    aligned_best = best["rlinf_openpi_pytorch_aligned"]
    best_reductions: dict[str, dict[str, float]] = {}
    for reference_name in (
        "openpi_jax",
        "official_openpi_pytorch_wrapper_baseline",
    ):
        if reference_name not in best:
            continue
        best_reductions[f"aligned_vs_{reference_name}"] = {
            metric: _reduction_percent(
                float(best[reference_name][metric]),
                float(aligned_best[metric]),
            )
            for metric in (
                "action_normalized_mse",
                "joint_velocity_mse",
                "gripper_position_mse",
                "joint_position_mse",
            )
        }
    return {
        "comparison": "pi05_droid wipe_board decoded action-chunk evaluation",
        "checkpoint_step_note": (
            "RLinf logs validation after update N at TensorBoard step N-1; rows "
            "report the corresponding checkpoint step N."
        ),
        "completion_requirements": {
            "expected_eval_count": expected_eval_count,
            "expected_num_chunks_per_eval": expected_num_chunks,
        },
        "runs": runs,
        "best_by_joint_position_mse": best,
        "final_evaluation": final,
        "aligned_best_reduction_percent": best_reductions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-metrics", type=Path, required=True)
    parser.add_argument("--aligned-rlinf-run", type=Path, required=True)
    parser.add_argument("--wrapper-baseline-run", type=Path)
    parser.add_argument("--expected-eval-count", type=int, default=10)
    parser.add_argument("--expected-num-chunks", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = build_comparison(
        args.jax_metrics,
        args.aligned_rlinf_run,
        args.wrapper_baseline_run,
        expected_eval_count=args.expected_eval_count,
        expected_num_chunks=args.expected_num_chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
