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

"""Tests for H200 OpenPI SFT smoke-test metric analysis."""

import importlib.util
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/pi05_droid_wipe_board/analyze_h200_smoke_test.py"
)
_SPEC = importlib.util.spec_from_file_location("analyze_h200_smoke_test", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _write_trial(trial_dir: Path, losses: list[float]) -> None:
    writer = SummaryWriter(trial_dir / "tensorboard")
    for step, loss in enumerate(losses):
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/learning_rate", 2.5e-5, step)
        writer.add_scalar("train/grad_norm", 1.0 + step, step)
        writer.add_scalar("time/training", 0.5, step)
    writer.close()


def test_load_trial_metrics_reports_tail_and_finite_values(tmp_path):
    trial_dir = tmp_path / "lr_2p5e_5"
    _write_trial(trial_dir, [3.0, 2.0, 1.0])

    summary = _MODULE.load_trial_metrics(trial_dir)

    assert summary["steps"] == 3
    assert summary["finite"] is True
    assert summary["tail_loss_mean"] == 2.0
    assert summary["grad_norm_max"] == 3.0


def test_summarize_root_selects_lowest_lr_tail_loss(tmp_path):
    root = tmp_path / "smoke"
    _write_trial(root / "lr_sweep/lr_1e_5", [2.0, 1.5])
    _write_trial(root / "lr_sweep/lr_2p5e_5", [1.0, 0.5])

    _MODULE.summarize_root(root)

    assert (root / "recommended_lr_trial.txt").read_text() == "lr_2p5e_5\n"
    summaries = json.loads((root / "summary.json").read_text())
    assert len(summaries) == 2
    assert (root / "summary.csv").is_file()
