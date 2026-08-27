# Copyright 2025 The RLinf Authors.
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

"""Static checks for the IsaacLab checkpoint evaluation launcher."""

import json
import logging
import subprocess
from pathlib import Path

import gymnasium as gym
import torch
from omegaconf import OmegaConf

from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.workers.env.env_worker import EnvWorker

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "starts" / "eval_isaaclab_checkpoint_videos.sh"


def test_checkpoint_eval_launcher_has_valid_bash_syntax() -> None:
    """The checkpoint evaluation launcher should pass Bash parsing."""
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_checkpoint_eval_launcher_help_describes_outputs() -> None:
    """Help should expose the checkpoint, video, and JSON contract."""
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CHECKPOINT" in result.stdout
    assert "external/table camera" in result.stdout
    assert "JSON file" in result.stdout


class _DummyVectorEnv(gym.Env):
    """Minimal vector env for testing deterministic video paths."""

    seed = 42
    num_envs = 2


def test_per_env_json_links_scenario_success_and_video(tmp_path: Path) -> None:
    """Each env result should retain its setting, outcome, and MP4 path."""
    video_cfg = OmegaConf.create(
        {
            "video_base_dir": str(tmp_path / "video"),
            "per_env_videos": True,
            "wait_for_video_writes": True,
        }
    )
    video_env = RecordVideo(_DummyVectorEnv(), video_cfg)

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "runner": {
                "ckpt_path": "/checkpoints/policy.pt",
                "eval_manifest_path": str(tmp_path / "eval_results.json"),
            },
            "env": {
                "eval": {
                    "per_episode_result_dir": str(tmp_path / "results"),
                    "per_episode_result_file_naming": "env_episode",
                }
            },
        }
    )
    worker._rank = 0
    worker.eval_env_list = [video_env]
    worker.eval_per_episode_result_dir = tmp_path / "results"
    worker.eval_per_episode_skip_existing = False
    env_info = {
        "scenario_id": torch.tensor([17, 23]),
        "success_once": torch.tensor([True, False]),
        "env_id": torch.tensor([0, 1]),
        "episode_id": torch.tensor([4, 4]),
        "worker_rank": torch.tensor([0, 0]),
    }

    worker._write_eval_per_episode_results(env_info, stage_id=0)

    result_paths = sorted((tmp_path / "results").glob("*.json"))
    assert len(result_paths) == 2
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    assert [result["setting"] for result in results] == [
        {"scenario_id": "000017"},
        {"scenario_id": "000023"},
    ]
    assert [result["policy_success"] for result in results] == [True, False]
    assert results[0]["video_path"].endswith("seed_42/0_env_000.mp4")
    assert results[1]["video_path"].endswith("seed_42/0_env_001.mp4")
    assert all(
        result["checkpoint_path"] == "/checkpoints/policy.pt" for result in results
    )
    for result in results:
        video_path = Path(result["video_path"])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()

    runner = object.__new__(EmbodiedEvalRunner)
    runner.cfg = worker.cfg
    runner.logger = logging.getLogger(__name__)
    runner._save_eval_manifest(
        {
            "worker_rank": torch.tensor([0, 0]),
            "stage_id": torch.tensor([0, 0]),
            "env_id": torch.tensor([0, 1]),
            "episode_id": torch.tensor([4, 4]),
        }
    )
    manifest = json.loads((tmp_path / "eval_results.json").read_text(encoding="utf-8"))
    assert manifest["num_results"] == 2
    assert [result["scenario_id"] for result in manifest["results"]] == [
        "000017",
        "000023",
    ]

    video_env.close()
