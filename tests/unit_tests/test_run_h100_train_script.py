# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the four-H100 pi0.5-DROID production training entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples/sft/pi05_droid_wipe_board/run_h100_train.sh"


def _dry_run_env(tmp_path: Path) -> dict[str, str]:
    """Build an isolated four-H100 environment for script dry runs."""
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    norm_stats = tmp_path / "norm_stats"
    fake_bin = tmp_path / "bin"
    dataset.mkdir()
    checkpoint.mkdir()
    norm_stats.mkdir()
    fake_bin.mkdir()
    (checkpoint / "model.safetensors").touch()
    (norm_stats / "norm_stats.json").write_text("{}\n")

    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"--query-gpu=name"* ]]; then\n'
        "  echo 'NVIDIA H100 80GB HBM3'\n"
        'elif [[ "$*" == *"--query-gpu=memory.total"* ]]; then\n'
        "  echo '81559'\n"
        "else\n"
        "  exit 2\n"
        "fi\n"
    )
    fake_nvidia_smi.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PYTHON_BIN": sys.executable,
            "DATASET_PATH": str(dataset),
            "BASE_CHECKPOINT_DIR": str(checkpoint),
            "NORM_STATS_DIR": str(norm_stats),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "H100_TRAIN_ROOT": str(tmp_path / "output"),
            "DRY_RUN": "1",
        }
    )
    return env


def test_h100_train_dry_run_uses_production_defaults(tmp_path: Path) -> None:
    """The default four-GPU recipe should require no accumulation."""
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        check=True,
        capture_output=True,
        env=_dry_run_env(tmp_path),
        text=True,
    )

    assert "Gradient accumulation:1" in result.stdout
    assert "cluster.component_placement.actor=0-3" in result.stdout
    assert "actor.micro_batch_size=16" in result.stdout
    assert "actor.global_batch_size=64" in result.stdout
    assert "actor.model.num_steps=10" in result.stdout
    assert "actor.optim.lr=5e-5" in result.stdout
    assert "actor.optim.lr_scheduler=openpi_cosine" in result.stdout
    assert "runner.val_check_interval=1000" in result.stdout
    assert not (tmp_path / "output").exists()


def test_h100_train_rejects_indivisible_global_batch(tmp_path: Path) -> None:
    """An invalid accumulation ratio should fail before training starts."""
    env = _dry_run_env(tmp_path)
    env["GLOBAL_BATCH_SIZE"] = "65"

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "must be divisible by micro batch x GPUs (64)" in result.stderr
