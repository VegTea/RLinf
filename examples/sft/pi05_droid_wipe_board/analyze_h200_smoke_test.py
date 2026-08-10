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

"""Validate and summarize TensorBoard metrics from the H200 SFT smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _find_event_file(trial_dir: Path) -> Path:
    event_files = sorted(trial_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found under {trial_dir}.")
    return event_files[-1]


def load_trial_metrics(trial_dir: Path) -> dict[str, Any]:
    """Load scalar metrics and return a compact trial summary."""
    accumulator = EventAccumulator(str(_find_event_file(trial_dir)))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if "train/loss" not in scalar_tags:
        raise KeyError(f"train/loss is missing from {trial_dir}.")

    def values(tag: str) -> list[float]:
        if tag not in scalar_tags:
            return []
        return [float(event.value) for event in accumulator.Scalars(tag)]

    losses = values("train/loss")
    learning_rates = values("train/learning_rate")
    grad_norms = values("train/grad_norm")
    training_times = values("time/training")
    finite = all(math.isfinite(value) for value in losses + grad_norms)
    window = min(10, len(losses))
    return {
        "trial": trial_dir.name,
        "steps": len(losses),
        "finite": finite,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "tail_loss_mean": statistics.fmean(losses[-window:]) if losses else None,
        "loss_min": min(losses) if losses else None,
        "learning_rate_max": max(learning_rates) if learning_rates else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "training_time_median_s": (
            statistics.median(training_times) if training_times else None
        ),
    }


def validate_trial(trial_dir: Path, min_steps: int) -> None:
    """Exit unsuccessfully when a trial has too few or non-finite updates."""
    summary = load_trial_metrics(trial_dir)
    print(json.dumps(summary, indent=2))
    if summary["steps"] < min_steps:
        raise RuntimeError(
            f"Trial emitted {summary['steps']} loss points; expected at least {min_steps}."
        )
    if not summary["finite"]:
        raise RuntimeError("Trial contains a non-finite loss or gradient norm.")


def summarize_root(root: Path) -> None:
    """Write JSON/CSV summaries and select the lowest finite LR tail loss."""
    summaries = []
    for phase in ("batch_probe", "lr_sweep"):
        phase_dir = root / phase
        if not phase_dir.is_dir():
            continue
        for trial_dir in sorted(path for path in phase_dir.iterdir() if path.is_dir()):
            try:
                summary = load_trial_metrics(trial_dir)
                summary["phase"] = phase
                summaries.append(summary)
            except (FileNotFoundError, KeyError) as error:
                summaries.append(
                    {
                        "trial": trial_dir.name,
                        "phase": phase,
                        "steps": 0,
                        "finite": False,
                        "error": str(error),
                    }
                )

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fieldnames = sorted({key for summary in summaries for key in summary})
    with (root / "summary.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    lr_trials = [
        summary
        for summary in summaries
        if summary.get("phase") == "lr_sweep"
        and summary.get("finite")
        and summary.get("tail_loss_mean") is not None
    ]
    if lr_trials:
        recommendation = min(lr_trials, key=lambda item: item["tail_loss_mean"])
        (root / "recommended_lr_trial.txt").write_text(f"{recommendation['trial']}\n")
        print(
            "Lowest finite 10-step tail loss: "
            f"{recommendation['trial']} ({recommendation['tail_loss_mean']:.8f})"
        )
    print(f"Wrote {root / 'summary.json'} and {root / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--validate-trial", type=Path)
    parser.add_argument("--min-steps", type=int, default=1)
    args = parser.parse_args()
    if (args.root is None) == (args.validate_trial is None):
        parser.error("Specify exactly one of --root or --validate-trial.")
    if args.validate_trial is not None:
        validate_trial(args.validate_trial, args.min_steps)
    else:
        summarize_root(args.root)


if __name__ == "__main__":
    main()
