#!/usr/bin/env python3
"""Plot curriculum range evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def load_rows(summary_csv: Path) -> list[dict]:
    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["stage_index"] = _int(row.get("stage_index"))
        row["threshold"] = _float(row.get("threshold"))
        row["scenario_count"] = _int(row.get("scenario_count"))
        row["expected_sample_count"] = _int(row.get("expected_sample_count"))
        row["num_trajectories"] = _int(row.get("num_trajectories"))
        row["success_once"] = _float(row.get("success_once"))
        row["return"] = _float(row.get("return"))
        row["episode_len"] = _float(row.get("episode_len"))
        row["reward"] = _float(row.get("reward"))
    return sorted(rows, key=lambda r: (r["stage_index"], r["mode"]))


def by_mode(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)
    for mode_rows in grouped.values():
        mode_rows.sort(key=lambda r: r["stage_index"])
    return grouped


def plot_success_by_stage(grouped: dict[str, list[dict]], output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for mode, rows in grouped.items():
        ax.plot(
            [r["stage_index"] for r in rows],
            [100.0 * r["success_once"] for r in rows],
            marker="o",
            linewidth=2,
            label=mode,
        )
    ax.set_title(f"{title} Success by Stage")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Success Once (%)")
    ax.set_xticks(range(0, 19))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_success_vs_count(grouped: dict[str, list[dict]], output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for mode, rows in grouped.items():
        ax.plot(
            [r["scenario_count"] for r in rows],
            [100.0 * r["success_once"] for r in rows],
            marker="o",
            linewidth=2,
            label=mode,
        )
    ax.set_title(f"{title} Success by Scenario Count")
    ax.set_xlabel("Scenario Count")
    ax.set_ylabel("Success Once (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_reward_by_stage(grouped: dict[str, list[dict]], output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for mode, rows in grouped.items():
        ax.plot(
            [r["stage_index"] for r in rows],
            [r["reward"] for r in rows],
            marker="o",
            linewidth=2,
            label=mode,
        )
    ax.set_title(f"{title} Reward by Stage")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Reward")
    ax.set_xticks(range(0, 19))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_counts(grouped: dict[str, list[dict]], output: Path, title: str) -> None:
    rows = grouped.get("cumulative") or next(iter(grouped.values()))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([r["stage_index"] for r in rows], [r["scenario_count"] for r in rows])
    ax.set_title(f"{title} Scenario Count by Stage")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Scenario Count")
    ax.set_xticks(range(0, 19))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_dashboard(grouped: dict[str, list[dict]], output: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for mode, rows in grouped.items():
        stages = [r["stage_index"] for r in rows]
        success = [100.0 * r["success_once"] for r in rows]
        counts = [r["scenario_count"] for r in rows]
        axes[0, 0].plot(stages, success, marker="o", linewidth=2, label=mode)
        axes[0, 1].plot(counts, success, marker="o", linewidth=2, label=mode)
        axes[1, 0].plot(stages, [r["reward"] for r in rows], marker="o", linewidth=2, label=mode)
        axes[1, 1].plot(stages, [r["episode_len"] for r in rows], marker="o", linewidth=2, label=mode)

    axes[0, 0].set_title("Success by Stage")
    axes[0, 0].set_xlabel("Stage")
    axes[0, 0].set_ylabel("Success Once (%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_xticks(range(0, 19))
    axes[0, 1].set_title("Success by Scenario Count")
    axes[0, 1].set_xlabel("Scenario Count")
    axes[0, 1].set_ylabel("Success Once (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[1, 0].set_title("Reward by Stage")
    axes[1, 0].set_xlabel("Stage")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].set_xticks(range(0, 19))
    axes[1, 1].set_title("Episode Length by Stage")
    axes[1, 1].set_xlabel("Stage")
    axes[1, 1].set_ylabel("Episode Length")
    axes[1, 1].set_xticks(range(0, 19))
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_compact(rows: list[dict], output: Path) -> None:
    fields = [
        "mode",
        "stage_index",
        "threshold",
        "scenario_count",
        "expected_sample_count",
        "num_trajectories",
        "success_once",
        "return",
        "episode_len",
        "reward",
        "status",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_report(rows: list[dict]) -> dict:
    grouped = by_mode(rows)
    report = {
        "num_rows": len(rows),
        "all_done": all(r.get("status") == "done" for r in rows),
        "modes": {},
    }
    for mode, mode_rows in grouped.items():
        final = max(mode_rows, key=lambda r: r["stage_index"])
        best = max(mode_rows, key=lambda r: r["success_once"])
        report["modes"][mode] = {
            "num_stages": len(mode_rows),
            "best_stage": best["stage_index"],
            "best_success_once": best["success_once"],
            "best_success_percent": 100.0 * best["success_once"],
            "final_stage": final["stage_index"],
            "final_success_once": final["success_once"],
            "final_success_percent": 100.0 * final["success_once"],
            "final_scenario_count": final["scenario_count"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    grouped = by_mode(rows)
    title = args.title or args.summary_csv.parent.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_success_by_stage(grouped, args.output_dir / "success_by_stage.png", title)
    plot_success_vs_count(grouped, args.output_dir / "success_by_scenario_count.png", title)
    plot_reward_by_stage(grouped, args.output_dir / "reward_by_stage.png", title)
    plot_counts(grouped, args.output_dir / "scenario_count_by_stage.png", title)
    plot_dashboard(grouped, args.output_dir / "dashboard.png", title)
    write_compact(rows, args.output_dir / "compact_results.csv")

    report = build_report(rows)
    report["summary_csv"] = str(args.summary_csv)
    report["output_dir"] = str(args.output_dir)
    with (args.output_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
