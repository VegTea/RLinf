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

"""Tests for the JAX/RLinf decoded SFT evaluation comparison."""

import importlib.util
import json
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/pi05_droid_wipe_board/compare_sft_evaluations.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_sft_evaluations", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

_METRICS = {
    "action_normalized_mse": 0.10,
    "action_normalized_mae": 0.20,
    "joint_velocity_mse": 0.30,
    "joint_velocity_mae": 0.40,
    "gripper_position_mse": 0.50,
    "gripper_position_mae": 0.60,
    "joint_position_mse": 0.70,
    "joint_position_mae": 0.80,
    "num_chunks": 200.0,
}


def _write_jax_metrics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "1000": {
                    **{f"eval/{name}": value for name, value in _METRICS.items()},
                    "eval/time_seconds": 2.0,
                }
            }
        )
    )


def _write_rlinf_metrics(run_dir: Path) -> None:
    writer = SummaryWriter(run_dir / "tensorboard")
    for name, value in _METRICS.items():
        writer.add_scalar(f"eval/{name}", value, 999)
    writer.add_scalar("time/evaluate", 3.0, 999)
    writer.add_scalar("train/loss", 0.05, 999)
    writer.close()


def test_load_rlinf_metrics_maps_logger_step_to_checkpoint(tmp_path):
    run_dir = tmp_path / "rlinf"
    _write_rlinf_metrics(run_dir)

    rows = _MODULE.load_rlinf_metrics(run_dir)

    assert rows[0]["checkpoint_step"] == 1000
    assert rows[0]["num_chunks"] == 200.0
    assert rows[0]["joint_position_mse"] == pytest.approx(
        _METRICS["joint_position_mse"]
    )
    assert rows[0]["train_loss"] == pytest.approx(0.05)


def test_build_comparison_keeps_wrapper_baseline_distinct(tmp_path):
    jax_path = tmp_path / "eval_metrics.json"
    aligned_run = tmp_path / "aligned"
    wrapper_run = tmp_path / "wrapper"
    _write_jax_metrics(jax_path)
    _write_rlinf_metrics(aligned_run)
    _write_rlinf_metrics(wrapper_run)

    result = _MODULE.build_comparison(jax_path, aligned_run, wrapper_run)

    assert set(result["runs"]) == {
        "openpi_jax",
        "rlinf_openpi_pytorch_aligned",
        "official_openpi_pytorch_wrapper_baseline",
    }
    assert (
        result["best_by_joint_position_mse"]["rlinf_openpi_pytorch_aligned"][
            "checkpoint_step"
        ]
        == 1000
    )
    assert result["final_evaluation"]["openpi_jax"]["checkpoint_step"] == 1000
    assert result["aligned_best_reduction_percent"]["aligned_vs_openpi_jax"][
        "joint_position_mse"
    ] == pytest.approx(0.0, abs=2e-6)


def test_completion_validation_rejects_partial_run():
    rows = [{"checkpoint_step": 1000, "num_chunks": 200.0}]

    with pytest.raises(ValueError, match="1 evaluation rows; expected 10"):
        _MODULE.validate_expected_evaluations(
            rows,
            expected_eval_count=10,
            expected_num_chunks=200,
            source="aligned",
        )


def test_completion_validation_rejects_wrong_chunk_count():
    rows = [{"checkpoint_step": 1000, "num_chunks": 20.0}]

    with pytest.raises(ValueError, match=r"checkpoints \[1000\]"):
        _MODULE.validate_expected_evaluations(
            rows,
            expected_eval_count=1,
            expected_num_chunks=200,
            source="aligned",
        )
