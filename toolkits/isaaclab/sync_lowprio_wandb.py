# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only live upload, optionally merging all attempts into one W&B run."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def stop_child(child: subprocess.Popen | None) -> None:
    """Bounded shutdown of only this supervisor's W&B client."""
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main_runs(attempts: list[Path]) -> list[tuple[Path, Path]]:
    """Find main streams in attempt order, never per-worker streams."""
    runs = []
    for attempt in sorted(attempts):
        files = list(attempt.glob("wandb/all/wandb/offline-run-*/run-*.wandb"))
        if len(files) > 1:
            raise RuntimeError(f"Multiple main runs in one attempt: {attempt}")
        if files and files[0].stat().st_size:
            runs.append((attempt, files[0]))
    return runs


def merged_identity(root: Path, entity: str, project: str, first_file: Path) -> str:
    """Persist the first main run's identity under the supervisor's exclusive lock."""
    path = root / "wandb_merged_run.json"
    if path.exists():
        identity = json.loads(path.read_text())
        if identity["entity"] != entity or identity["project"] != project:
            raise ValueError(
                "Merged W&B entity/project differs from the stored identity"
            )
        return identity["run_id"]
    run_id = first_file.stem.removeprefix("run-")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "entity": entity,
                "project": project,
                "run_id": run_id,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(path)
    return run_id


def closed_fragment(attempt: Path, entity: str, project: str) -> Path | None:
    """Export persisted main scalars; preempted raw .wandb tails may be corrupt."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    complete = attempt / "wandb_closed_fragment.json"
    if complete.exists():
        return Path(json.loads(complete.read_text())["path"])
    rows = {}
    for file in sorted((attempt / "tensorboard/all").glob("events.out*")):
        events = EventAccumulator(str(file), size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            for event in events.Scalars(tag):
                rows.setdefault(event.step, {})[tag] = event.value
    if not rows:
        return None
    import yaml

    import wandb

    output = attempt / "wandb_closed_fragment"
    output.mkdir(exist_ok=True)
    config_file = attempt / "tensorboard/all/config.yaml"
    config = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    name = (
        config.get("runner", {})
        .get("logger", {})
        .get("experiment_name", attempt.parent.parent.name)
    )
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        dir=str(output),
        mode="offline",
        reinit="create_new",
        settings=wandb.Settings(console="off", x_disable_stats=True),
    )
    file = Path(run.dir).parent / f"run-{run.id}.wandb"
    for step, values in sorted(rows.items()):
        run.log(values, step=step, commit=True)
    run.finish()
    complete.write_text(json.dumps({"path": str(file), "steps": sorted(rows)}) + "\n")
    return file


def main() -> None:
    """Follow attempts; finish old fragments before live-syncing the current one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--entity", default="vegtea-hust")
    parser.add_argument("--project", default="rlinf")
    parser.add_argument(
        "--merge-attempts",
        action="store_true",
        help="Backfill main streams into the first run and keep its ID across restarts.",
    )
    parser.add_argument(
        "--wandb-bin",
        default=str(Path(__file__).resolve().parents[2] / ".venv-model312/bin/wandb"),
    )
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    child = None
    target = None
    run_id = None
    log_offset = 0
    with (root / "wandb_sync.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "wandb_sync_supervisor.pid").write_text(f"{os.getpid()}\n")
        try:
            while True:
                pointer = root / "active_attempt"
                current = (
                    Path(pointer.read_text().strip()).resolve()
                    if pointer.exists()
                    else None
                )
                if current is not None and current.parent != root / "attempts":
                    raise ValueError(
                        "active_attempt must name a direct child of run_root/attempts"
                    )
                if current is None:
                    time.sleep(5)
                    continue
                attempts = (
                    [
                        p
                        for p in (root / "attempts").glob("*")
                        if p.is_dir() and not p.is_symlink()
                    ]
                    if args.merge_attempts
                    else ([current] if current else [])
                )
                runs = main_runs(attempts)
                if args.merge_attempts and runs and run_id is None:
                    run_id = merged_identity(
                        root, args.entity, args.project, runs[0][1]
                    )
                    print(
                        f"Unified run: https://wandb.ai/{args.entity}/{args.project}/runs/{run_id}",
                        flush=True,
                    )
                marker = f"wandb_sync.{run_id}.ok" if run_id else "wandb_sync.ok"
                if child is not None and child.poll() is not None:
                    with (target[0] / "wandb_sync.log").open("rb") as output:
                        output.seek(log_offset)
                        failed = b"wandb: ERROR" in output.read()
                    # W&B 0.28.2 can print a sync error yet exit with status zero.
                    if child.returncode == 0 and not failed:
                        (target[0] / marker).touch()
                    else:
                        print(
                            f"W&B exited {child.returncode}; retrying {target[0]}",
                            flush=True,
                        )
                    child = None
                    target = None
                pending = [(a, f) for a, f in runs if not (a / marker).exists()]
                wanted = (
                    (
                        pending[0][0],
                        pending[0][1],
                        pending[0][0] == current
                        and not (pending[0][0] / "exit_status").exists()
                        and not (pending[0][0] / "wandb_source_closed").exists(),
                    )
                    if pending
                    else None
                )
                if wanted != target:
                    stop_child(child)
                    child = None
                    target = wanted
                if target is not None and child is None:
                    attempt, file, live = target
                    if run_id and not live:
                        file = closed_fragment(attempt, args.entity, args.project)
                        if file is None:
                            (attempt / marker).touch()
                            target = None
                            continue
                    command = [
                        args.wandb_bin,
                        "beta",
                        "sync",
                        "--yes",
                        "--entity",
                        args.entity,
                        "--project",
                        args.project,
                    ]
                    if run_id:
                        # Previous per-attempt uploads must not suppress backfill.
                        command += ["--id", run_id, "--no-skip-synced"]
                    if live:
                        command.append("--live")
                    command.append(str(file))
                    with (attempt / "wandb_sync.log").open("a") as output:
                        log_offset = output.tell()
                        child = subprocess.Popen(
                            command,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "WANDB_MODE": "online"},
                        )
                    (attempt / "wandb_sync.pid").write_text(f"{child.pid}\n")
                    print(
                        f"{'Live' if live else 'Final'} sync {attempt.name}: pid={child.pid}",
                        flush=True,
                    )
                time.sleep(5)
        except KeyboardInterrupt:
            pass
        finally:
            stop_child(child)


if __name__ == "__main__":
    main()
