# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for four-GPU configuration and restart-aware W&B supervision."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from toolkits.isaaclab.sync_lowprio_wandb import (
    closed_fragment,
    main_runs,
    merged_identity,
)

REPO = Path(__file__).resolve().parents[2]


def write_scalar(attempt, step):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(attempt / "tensorboard/all"))
    writer.add_event(
        Event(
            wall_time=time.time(),
            step=step,
            summary=Summary(
                value=[Summary.Value(tag="env/success_once", simple_value=0.5)]
            ),
        )
    )
    writer.close()


def test_lowprio_config(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(REPO / "examples/embodiment/config"), version_base=None
    ):
        cfg = compose(
            config_name="isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100_instanceable_safety_lowprio_4gpu"
        )
    assert cfg.env.train.total_num_envs == 64
    assert cfg.actor.global_batch_size == 4 * cfg.actor.micro_batch_size
    assert cfg.runner.save_interval == 1
    assert cfg.runner.save_best_checkpoint
    assert cfg.runner.keep_last_n_checkpoints == 2
    assert cfg.runner.resume_dir == "auto"
    subprocess.run(
        [
            "bash",
            "-n",
            str(REPO / "starts/train_h100_4x_nearest100_safety_lowprio_isaacsim6.sh"),
        ],
        check=True,
    )


@pytest.mark.parametrize("merge", [False, True])
def test_sync_follows_restart(tmp_path, merge):
    fake = tmp_path / "fake-wandb"
    fake.write_text(
        f"#!{sys.executable}\nimport time, sys, json\n"
        "print(json.dumps(sys.argv[1:]), flush=True)\n"
        "if '--live' in sys.argv: time.sleep(120)\n"
    )
    fake.chmod(0o755)
    supervisor = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "toolkits/isaaclab/sync_lowprio_wandb.py"),
            str(tmp_path),
            "--wandb-bin",
            str(fake),
        ]
        + (["--merge-attempts"] if merge else [])
    )
    children = []
    try:
        for name in ("first", "restart"):
            attempt = tmp_path / "attempts" / name
            run = attempt / "wandb/all/wandb/offline-run-test"
            run.mkdir(parents=True)
            (run / f"run-{name}.wandb").write_bytes(b"fake")
            if merge:
                write_scalar(attempt, 0 if name == "first" else 1)
            pointer = tmp_path / ".active.tmp"
            pointer.write_text(str(attempt))
            pointer.replace(tmp_path / "active_attempt")
            deadline = time.monotonic() + 25
            while not (attempt / "wandb_sync.pid").exists():
                assert supervisor.poll() is None
                assert time.monotonic() < deadline, (
                    "W&B supervisor did not follow attempt"
                )
                time.sleep(0.1)
            children.append(int((attempt / "wandb_sync.pid").read_text()))
        assert children[0] != children[1]
        if merge:
            identity = json.loads((tmp_path / "wandb_merged_run.json").read_text())
            assert identity["run_id"] == "first"
            first = tmp_path / "attempts/first"
            assert (first / "wandb_sync.first.ok").exists()
            final_command = json.loads(
                (first / "wandb_sync.log").read_text().splitlines()[-1]
            )
            assert "--live" not in final_command
            assert final_command[final_command.index("--id") + 1] == "first"
            assert "--no-skip-synced" in final_command
            assert not (tmp_path / "attempts/restart/wandb_sync.first.ok").exists()
        try:
            os.kill(children[0], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("Old live-sync client was left running after restart")
        assert (
            int((tmp_path / "wandb_sync_supervisor.pid").read_text()) == supervisor.pid
        )
    finally:
        supervisor.terminate()
        supervisor.wait(timeout=15)
    for pid in children:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("W&B client leaked after supervisor shutdown")


def test_sync_error_with_zero_exit_is_not_marked(tmp_path):
    fake = tmp_path / "fake-wandb"
    fake.write_text(
        f"#!{sys.executable}\nprint('wandb: ERROR unexpected EOF', flush=True)\n"
    )
    fake.chmod(0o755)
    attempt = tmp_path / "attempts/first"
    stream = attempt / "wandb/all/wandb/offline-run-first"
    stream.mkdir(parents=True)
    (stream / "run-first.wandb").write_bytes(b"fake")
    (tmp_path / "active_attempt").write_text(str(attempt))
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "toolkits/isaaclab/sync_lowprio_wandb.py"),
            str(tmp_path),
            "--wandb-bin",
            str(fake),
        ]
    )
    try:
        deadline = time.monotonic() + 20
        output = attempt / "wandb_sync.log"
        while not output.exists() or output.read_text().count("wandb: ERROR") < 2:
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.1)
        assert not (attempt / "wandb_sync.ok").exists()
    finally:
        process.terminate()
        process.wait(timeout=15)


def test_closed_fragment_has_committed_history(tmp_path):
    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore

    assert closed_fragment(tmp_path, "entity", "project") is None
    write_scalar(tmp_path, 58)
    file = closed_fragment(tmp_path, "entity", "project")
    assert closed_fragment(tmp_path, "entity", "project") == file
    reader = DataStore()
    reader.open_for_scan(str(file))
    histories = []
    has_exit = False
    while (data := reader.scan_data()) is not None:
        record = Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") == "history":
            histories.append(
                {
                    item.key or ".".join(item.nested_key): json.loads(item.value_json)
                    for item in record.history.item
                }
            )
        has_exit |= record.WhichOneof("record_type") == "exit"
    assert has_exit and len(histories) == 1
    assert histories[0]["_step"] == 58
    assert histories[0]["env/success_once"] == 0.5


def test_merge_identity_and_main_stream_isolation(tmp_path):
    attempt = tmp_path / "attempts/first"
    run = attempt / "wandb/all/wandb/offline-run-1"
    run.mkdir(parents=True)
    first_file = run / "run-first.wandb"
    first_file.write_bytes(b"data")
    worker = attempt / "worker_logs/ActorGroup/rank_0/wandb/wandb/offline-run-1"
    worker.mkdir(parents=True)
    (worker / "run-worker.wandb").write_bytes(b"data")
    assert main_runs([attempt]) == [(attempt, first_file)]
    assert merged_identity(tmp_path, "entity", "project", first_file) == "first"
    assert (
        merged_identity(tmp_path, "entity", "project", Path("run-later.wandb"))
        == "first"
    )
    with pytest.raises(ValueError, match="entity/project"):
        merged_identity(tmp_path, "entity", "different", first_file)
    (run / "run-duplicate.wandb").write_bytes(b"data")
    with pytest.raises(RuntimeError, match="Multiple main runs"):
        main_runs([attempt])
