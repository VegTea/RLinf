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

import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner


def make_runner(root: Path) -> EmbodiedRunner:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"runner": {"logger": {"log_path": str(root), "experiment_name": "test"}}}
    )
    runner.actor = MagicMock()
    runner.logger = MagicMock()
    runner.global_step = 0
    runner.save_best_checkpoint = True
    runner.best_checkpoint = None
    runner._owned_best_checkpoint = None
    runner.keep_last_n_checkpoints = 0
    return runner


def test_best_retention_periodic_metadata_and_resume(tmp_path):
    runner = make_runner(tmp_path / "original")
    runner._maybe_save_best_checkpoint(0.2)
    first = Path(runner.best_checkpoint["path"])
    assert first.name == "global_step_0"
    assert (first / "actor").is_dir()
    runner.global_step = 1
    runner._save_checkpoint()
    periodic = first.parent.parent / "global_step_1"
    assert (
        json.loads((periodic / "runner_state.json").read_text())["best_checkpoint"][
            "score"
        ]
        == 0.2
    )

    for score in (None, float("nan"), float("inf"), -float("inf"), 0.1, 0.2):
        runner._maybe_save_best_checkpoint(score)
    assert runner.actor.save_checkpoint.call_count == 2
    runner._maybe_save_best_checkpoint(0.7)
    best = Path(runner.best_checkpoint["path"])
    assert not first.exists()
    assert periodic.exists()
    assert list(best.parent.glob("global_step_*")) == [best]
    assert json.loads((best.parent / "best.json").read_text()) == runner.best_checkpoint

    resumed = make_runner(tmp_path / "resumed")
    # Even an older periodic snapshot restores the source run's retained best.
    resumed._restore_best_checkpoint(periodic)
    assert resumed.best_checkpoint == runner.best_checkpoint
    resumed._maybe_save_best_checkpoint(0.6)
    resumed.actor.save_checkpoint.assert_not_called()
    resumed.global_step = 2
    resumed._maybe_save_best_checkpoint(0.8)
    assert best.exists(), "A new run must never delete its resume source's best"
    resumed_again = make_runner(tmp_path / "resumed-again")
    resumed_again._restore_best_checkpoint(Path(resumed.best_checkpoint["path"]))
    assert resumed_again.best_checkpoint["score"] == 0.8

    legacy = make_runner(tmp_path / "legacy")
    legacy._restore_best_checkpoint(tmp_path / "global_step_520")
    assert legacy.best_checkpoint is None


def test_best_save_failure_preserves_previous(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    runner._maybe_save_best_checkpoint(0.2)
    previous = runner.best_checkpoint.copy()
    best_dir = Path(previous["path"]).parent
    index = (best_dir / "best.json").read_bytes()
    runner.global_step = 1
    runner.actor.save_checkpoint.return_value.wait.side_effect = RuntimeError(
        "disk full"
    )
    with pytest.raises(RuntimeError, match="disk full"):
        runner._maybe_save_best_checkpoint(0.9)
    assert runner.best_checkpoint == previous
    assert (best_dir / "best.json").read_bytes() == index
    assert sorted(p.name for p in best_dir.iterdir()) == ["best.json", "global_step_0"]

    runner.actor.save_checkpoint.return_value.wait.side_effect = None
    original_replace = Path.replace

    def fail_index_publish(path, target):
        if path.name == ".best.json.tmp":
            raise OSError("publish failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_index_publish)
    with pytest.raises(OSError, match="publish failed"):
        runner._maybe_save_best_checkpoint(0.9)
    assert runner.best_checkpoint == previous
    assert (best_dir / "best.json").read_bytes() == index
    assert sorted(p.name for p in best_dir.iterdir()) == ["best.json", "global_step_0"]


def test_existing_checkpoint_not_overwritten(tmp_path):
    runner = make_runner(tmp_path)
    destination = tmp_path / "test/checkpoints/best/global_step_0"
    destination.mkdir(parents=True)
    marker = destination / "user-file"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        runner._maybe_save_best_checkpoint(0.5)
    assert marker.read_text() == "keep"
    runner.actor.save_checkpoint.assert_not_called()


def test_best_requires_synchronized_policy():
    cfg = OmegaConf.create(
        {"runner": {"save_best_checkpoint": True, "weight_sync_interval": 2}}
    )
    with pytest.raises(ValueError, match="weight_sync_interval=1"):
        EmbodiedRunner(cfg, None, None, None)


@pytest.mark.parametrize("enabled", [False, True])
def test_run_saves_measured_policy_before_update(tmp_path, monkeypatch, enabled):
    runner = make_runner(tmp_path)
    runner.save_best_checkpoint = enabled
    runner.cfg.runner.save_interval = 1
    runner.cfg.runner.val_check_interval = -1
    runner.max_steps = 1
    runner.weight_sync_interval = 1
    runner.need_curriculum = False
    runner.scenario_curriculum_enabled = False
    runner.train_set_suc_rate = 0
    runner.reward = None
    runner.env = MagicMock()
    runner.rollout = MagicMock()
    runner.env_channel = runner.rollout_channel = runner.actor_channel = None
    runner.reward_channel = None
    runner.metric_logger = MagicMock()
    runner.timer = MagicMock(side_effect=lambda _: nullcontext())
    runner.timer.consume_durations.return_value = {}
    runner.log_queue = MagicMock()
    runner.log_thread = MagicMock()
    runner.print_metrics_table_async = MagicMock()
    runner._log_ranked_metrics = MagicMock()
    runner._compute_scenario_group_metrics = MagicMock(return_value={})
    runner._maybe_update_scenario_curriculum = MagicMock(return_value={})
    runner.actor.compute_advantages_and_returns.return_value.wait.return_value = []
    runner.actor.run_training.return_value.wait.return_value = []
    measured_metrics = {"success_once": 0.6, "success_rate": 0.25}
    runner.env.interact.return_value.wait.return_value = [measured_metrics]
    for handle in (
        runner.env.interact.return_value,
        runner.rollout.generate.return_value,
        runner.actor.run_training.return_value,
    ):
        handle.consume_durations.return_value = ({}, [])
    monkeypatch.setattr(
        "rlinf.runners.embodied_runner.compute_evaluate_metrics",
        lambda _: measured_metrics,
    )
    events = []
    runner.update_rollout_weights = lambda: events.append("sync")
    runner.env.interact.return_value.wait.side_effect = lambda: (
        events.append("measured") or [measured_metrics]
    )
    runner.actor.save_checkpoint.side_effect = lambda path, step: (
        events.append(("save", step, path)) or MagicMock()
    )
    runner.actor.run_training.return_value.wait.side_effect = lambda: (
        events.append("train") or []
    )
    runner.run()
    assert events[0:2] == ["sync", "measured"]
    if enabled:
        assert events[2][0:2] == ("save", 0)
        assert events[3] == "train"
        assert events[4][0:2] == ("save", 1)
        assert runner.best_checkpoint["step"] == 0
        assert runner.best_checkpoint["metric"] == "env/success_rate"
        assert runner.best_checkpoint["score"] == 0.25
    else:
        assert events[2] == "train"
        assert events[3][0:2] == ("save", 1)
        assert runner.best_checkpoint is None
        assert not list(tmp_path.rglob("runner_state.json"))
        assert not list(tmp_path.rglob("best"))
    assert runner.global_step == 1


def test_init_workers_resumes_best_step(tmp_path):
    runner = make_runner(tmp_path)
    runner.global_step = 520
    runner._maybe_save_best_checkpoint(0.5)
    resumed = make_runner(tmp_path / "new")
    resumed.cfg.runner.resume_dir = runner.best_checkpoint["path"] + "/"
    resumed.rollout = MagicMock()
    resumed.env = MagicMock()
    resumed.reward = None
    resumed.scenario_curriculum_enabled = False
    resumed.init_workers()
    assert resumed.global_step == 520
    assert resumed.best_checkpoint["score"] == 0.5
    resumed.actor.load_checkpoint.assert_called_once_with(
        str(Path(runner.best_checkpoint["path"]) / "actor")
    )


def test_periodic_retention_survives_restart_and_protects_best(tmp_path):
    runner = make_runner(tmp_path / "attempt1")
    runner.cfg.runner.checkpoint_dir = str(tmp_path / "persistent")
    runner.keep_last_n_checkpoints = 2
    runner._maybe_save_best_checkpoint(0.5)
    best = Path(runner.best_checkpoint["path"])
    for step in (1, 2):
        runner.global_step = step
        runner._save_checkpoint()
    root = runner.checkpoint_root
    # An interrupted save, a legacy directory, and a symlink are never owned.
    (root / ".saving-interrupted").mkdir()
    (root / "global_step_999").mkdir()
    (root / "global_step_999" / "actor").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep").write_text("keep")
    (root / "global_step_998").symlink_to(external, target_is_directory=True)

    resumed = make_runner(tmp_path / "attempt2")
    resumed.cfg.runner.checkpoint_dir = str(root)
    resumed.cfg.runner.resume_dir = "auto"
    resumed.keep_last_n_checkpoints = 2
    resumed.rollout = MagicMock()
    resumed.env = MagicMock()
    resumed.reward = None
    resumed.scenario_curriculum_enabled = False
    resumed.init_workers()
    assert resumed.global_step == 2
    resumed.actor.load_checkpoint.assert_called_once_with(
        str(root / "global_step_2/actor")
    )
    assert resumed.best_checkpoint["score"] == 0.5
    assert resumed._owned_best_checkpoint == best
    resumed.global_step = 3
    resumed._save_checkpoint()
    assert not (root / "global_step_1").exists()
    assert [p.name for p in resumed._complete_checkpoints(root)] == [
        "global_step_2",
        "global_step_3",
    ]
    assert best.is_dir()
    assert (external / "keep").read_text() == "keep"
    assert (root / "global_step_999").is_dir()
    assert (root / ".saving-interrupted").is_dir()

    # Failed actor save cannot evict either retained checkpoint.
    resumed.global_step = 4
    resumed.actor.save_checkpoint.return_value.wait.side_effect = RuntimeError(
        "preempted"
    )
    with pytest.raises(RuntimeError, match="preempted"):
        resumed._save_checkpoint()
    assert not (root / "global_step_4").exists()
    assert len(resumed._complete_checkpoints(root)) == 2
    assert list(root.glob(".saving-*")) == [root / ".saving-interrupted"]
    resumed.actor.save_checkpoint.return_value.wait.side_effect = None
    resumed._maybe_save_best_checkpoint(0.8)
    assert not best.exists()
    assert len(resumed._complete_checkpoints(root / "best")) == 1
    assert len(resumed._complete_checkpoints(root)) == 2


def test_auto_resume_best_fallback_and_empty_start(tmp_path):
    runner = make_runner(tmp_path)
    runner.cfg.runner.resume_dir = "auto"
    runner.rollout = MagicMock()
    runner.env = MagicMock()
    runner.reward = None
    runner.scenario_curriculum_enabled = False
    runner.init_workers()
    runner.actor.load_checkpoint.assert_not_called()
    runner._maybe_save_best_checkpoint(0.4)
    path = runner.best_checkpoint["path"]
    runner.init_workers()
    runner.actor.load_checkpoint.assert_called_once_with(str(Path(path) / "actor"))
    # First rollout can improve the score of the same resumed initial policy.
    runner._maybe_save_best_checkpoint(0.5)
    assert runner.actor.save_checkpoint.call_count == 1
    assert json.loads((Path(path).parent / "best.json").read_text())["score"] == 0.5


def test_periodic_publish_failure_preserves_previous(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    runner.keep_last_n_checkpoints = 2
    for step in (1, 2):
        runner.global_step = step
        runner._save_checkpoint()
    root = runner.checkpoint_root
    runner.global_step = 3
    with monkeypatch.context() as patcher:
        patcher.setattr(
            Path, "rename", MagicMock(side_effect=OSError("publish failed"))
        )
        with pytest.raises(OSError, match="publish failed"):
            runner._save_checkpoint()
    assert [p.name for p in runner._complete_checkpoints(root)] == [
        "global_step_1",
        "global_step_2",
    ]
    assert not list(root.glob(".saving-*"))
    runner.global_step = 2
    with pytest.raises(FileExistsError):
        runner._save_checkpoint()


def test_resume_recovers_best_published_before_index(tmp_path):
    runner = make_runner(tmp_path)
    runner._maybe_save_best_checkpoint(0.4)
    previous = Path(runner.best_checkpoint["path"])
    runner.global_step = 1
    runner._save_checkpoint()
    orphan = previous.parent / "global_step_1"
    runner._save_checkpoint(str(orphan))
    candidate = {
        "metric": "env/success_rate",
        "score": 0.8,
        "step": 1,
        "path": str(orphan),
    }
    (orphan / "runner_state.json").write_text(
        json.dumps({"best_checkpoint": candidate})
    )
    # The process died before replacing best.json or removing the old best.
    runner.best_checkpoint = None
    runner._owned_best_checkpoint = None
    runner._restore_best_checkpoint(runner.checkpoint_root / "global_step_1")
    assert runner.best_checkpoint == candidate
    assert json.loads((orphan.parent / "best.json").read_text()) == candidate
    assert not previous.exists()
    assert orphan.exists()


@pytest.mark.parametrize("external_source", [False, True])
def test_legacy_metric_is_not_compared_and_weights_survive_until_new_best(
    tmp_path, external_source
):
    original = make_runner(tmp_path / "source")
    original._maybe_save_best_checkpoint(0.95)
    legacy = dict(original.best_checkpoint, metric="env/success_once")
    old_path = Path(legacy["path"])
    (old_path / "runner_state.json").write_text(json.dumps({"best_checkpoint": legacy}))
    (old_path.parent / "best.json").write_text(json.dumps(legacy))
    original.best_checkpoint = legacy
    original.global_step = 1
    original._save_checkpoint()
    periodic = original.checkpoint_root / "global_step_1"

    resumed = make_runner(tmp_path / ("new" if external_source else "source"))
    resumed.cfg.runner.resume_dir = str(periodic)
    resumed.rollout = MagicMock()
    resumed.env = MagicMock()
    resumed.reward = None
    resumed.scenario_curriculum_enabled = False
    resumed.init_workers()
    assert resumed.global_step == 1
    resumed.actor.load_checkpoint.assert_called_once_with(str(periodic / "actor"))
    assert resumed.best_checkpoint is None
    assert old_path.exists()
    resumed.actor.save_checkpoint.return_value.wait.side_effect = RuntimeError(
        "disk full"
    )
    with pytest.raises(RuntimeError, match="disk full"):
        resumed._maybe_save_best_checkpoint(0.2)
    assert old_path.exists()
    assert (
        json.loads((old_path.parent / "best.json").read_text())["metric"]
        == "env/success_once"
    )
    resumed.actor.save_checkpoint.return_value.wait.side_effect = None
    resumed._maybe_save_best_checkpoint(0.2)
    assert resumed.best_checkpoint["metric"] == "env/success_rate"
    assert resumed.best_checkpoint["score"] == 0.2
    assert old_path.exists() == external_source
    again = make_runner(tmp_path / "another")
    again._restore_best_checkpoint(Path(resumed.best_checkpoint["path"]))
    assert again.best_checkpoint == resumed.best_checkpoint


def test_legacy_best_fallback_reuses_same_step_weights(tmp_path):
    runner = make_runner(tmp_path)
    runner._maybe_save_best_checkpoint(0.95)
    legacy = dict(runner.best_checkpoint, metric="env/success_once")
    path = Path(legacy["path"])
    (path / "runner_state.json").write_text(json.dumps({"best_checkpoint": legacy}))
    (path.parent / "best.json").write_text(json.dumps(legacy))
    runner._restore_best_checkpoint(path)
    assert runner.best_checkpoint is None
    runner._maybe_save_best_checkpoint(0.2)
    assert runner.actor.save_checkpoint.call_count == 1
    runner._restore_best_checkpoint(path)
    assert runner.best_checkpoint["metric"] == "env/success_rate"
    assert runner.best_checkpoint["score"] == 0.2


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_invalid_retention_rejected(value):
    cfg = OmegaConf.create(
        {
            "runner": {
                "weight_sync_interval": 1,
                "keep_last_n_checkpoints": value,
            }
        }
    )
    with pytest.raises(ValueError, match="keep_last_n_checkpoints"):
        EmbodiedRunner(cfg, None, None, None)
