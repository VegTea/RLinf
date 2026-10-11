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

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# The low-priority entrypoint has a separate run-root CLI and dedicated tests.
LAUNCHERS = sorted(
    path
    for path in (REPO / "starts").glob("train*.sh")
    if path.name != "train_h100_4x_nearest100_safety_lowprio_isaacsim6.sh"
)


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + text)
    path.chmod(0o755)


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda p: p.stem)
@pytest.mark.parametrize("wandb", [False, True])
def test_launchers_forward_options_offline(tmp_path, launcher, wandb):
    """Run real launchers against fake runtimes; never start Ray or training."""
    starts = tmp_path / "starts"
    starts.mkdir()
    shutil.copy(launcher, starts / launcher.name)
    shutil.copy(REPO / "starts/_train_options.sh", starts / "_train_options.sh")
    (starts / "source.sh").write_text(
        'REPO_PATH="$(dirname "$STARTS_DIR")"\n'
        'export RLINF_MODEL_PYTHON="$REPO_PATH/.venv-model312/bin/python"\n'
        'cd "$REPO_PATH"\n'
    )
    for venv in (".venv", ".venv-model312", ".venv-isaacsim6"):
        executable(tmp_path / venv / "bin/python", "echo 8\n")
        (tmp_path / venv / "bin/activate").write_text(":\n")
    executable(tmp_path / "bin/ray", "exit 0\n")
    executable(tmp_path / "bin/nvidia-smi", 'printf "GPU\\n%.0s" {1..8}\n')
    executable(
        tmp_path / "examples/embodiment/run_embodiment.sh",
        'printf "%s\\0" "$WANDB_MODE" "$@"\n',
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CUDA_VISIBLE_DEVICES", "SAVE_INTERVAL", "USE_WANDB", "MAX_EPOCHS")
    }
    env.update(PATH=f"{tmp_path}/bin:{env['PATH']}", WANDB_MODE="online")
    if wandb:
        env.update(USE_WANDB="true", SAVE_INTERVAL="17")
    override = "runner.save_interval=23"
    # W&B chosen through CLI must also remain offline with USE_WANDB=false.
    backend_override = "runner.logger.logger_backends=[tensorboard,wandb]"
    result = subprocess.run(
        ["bash", str(starts / launcher.name), override, backend_override],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    # Some launchers print diagnostics before exec, so inspect the first NUL item.
    args = result.stdout.decode().split("\0")
    assert args[0].endswith("offline")
    assert f"runner.save_interval={17 if wandb else 100}" in args
    expected = "[tensorboard,wandb]" if wandb else "[tensorboard]"
    assert f"runner.logger.logger_backends={expected}" in args
    assert "++runner.save_best_checkpoint=true" in args
    assert args[-3:] == [override, backend_override, ""]


@pytest.mark.parametrize(
    "variable,value",
    [("SAVE_INTERVAL", "0"), ("SAVE_INTERVAL", "abc"), ("USE_WANDB", "maybe")],
)
def test_invalid_train_options_fail(variable, value):
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", str(REPO / "starts/_train_options.sh")],
        env={
            **os.environ,
            "SAVE_INTERVAL": "40",
            "USE_WANDB": "false",
            variable: value,
        },
        capture_output=True,
    )
    assert result.returncode == 2
    assert variable in result.stderr.decode()


def test_periodic_saving_can_be_disabled():
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "${TRAIN_OPTIONS[@]}"',
            "bash",
            str(REPO / "starts/_train_options.sh"),
        ],
        env={**os.environ, "SAVE_INTERVAL": "-1", "USE_WANDB": "0"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "runner.save_interval=-1" in result.stdout
    assert "++runner.save_best_checkpoint=true" in result.stdout
