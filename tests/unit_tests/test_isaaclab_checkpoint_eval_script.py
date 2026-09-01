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

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv
from rlinf.envs.isaaclab.scenario_export import validate_scenario_file
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
    assert "--scenario-file" in result.stdout
    assert "eval_scenarios.jsonl" in result.stdout
    assert "eval_throughput.json" in result.stdout


class _DummyVectorEnv(gym.Env):
    """Minimal vector env for testing deterministic video paths."""

    seed = 42
    num_envs = 2

    def get_episode_scenario_record(
        self, env_id: int, episode_id: int, *, pop: bool = False
    ) -> dict:
        """Return a complete reloadable scenario for test aggregation."""
        return {
            "id": f"source_{env_id}",
            "_source_scenario_id": f"source_{env_id}",
            "table_asset": "/assets/table.usd",
            "cube_1_pos": [0.4, 0.0, 0.02],
            "cube_1_rpy": [0.0, 0.0, 0.0],
            "cube_2_pos": [0.5, 0.0, 0.02],
            "cube_2_rpy": [0.0, 0.0, 0.0],
            "cube_3_pos": [0.6, 0.0, 0.02],
            "cube_3_rpy": [0.0, 0.0, 0.0],
            "table_cam_pos": [1.0, 0.0, 0.4],
            "table_cam_rot": [1.0, 0.0, 0.0, 0.0],
        }


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
                "eval_scenario_output_path": str(
                    tmp_path / "eval_scenarios.jsonl"
                ),
            },
            "env": {
                "eval": {
                    "per_episode_result_dir": str(tmp_path / "results"),
                    "per_episode_result_file_naming": "env_episode",
                    "total_num_envs": 2,
                    "video_cfg": {
                        "video_base_dir": str(tmp_path / "video"),
                    },
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
        "first_success_step": torch.tensor([200, -1]),
        "episode_len": torch.tensor([450, 450]),
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
    assert [result["first_success_step"] for result in results] == [200, -1]
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
    assert manifest["num_success"] == 1
    assert manifest["num_failed"] == 1
    assert manifest["success_rate"] == 0.5
    assert manifest["successful_results"] == ["seed_42/0_env_000.mp4"]
    assert manifest["failed_results"] == ["seed_42/0_env_001.mp4"]
    assert set(manifest) == {
        "num_results",
        "num_success",
        "num_failed",
        "success_rate",
        "successful_results",
        "failed_results",
    }
    scenarios = [
        json.loads(line)
        for line in (tmp_path / "eval_scenarios.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [scenario["id"] for scenario in scenarios] == [
        "eval_000000",
        "eval_000001",
    ]
    assert [scenario["source_scenario_id"] for scenario in scenarios] == [
        "source_0",
        "source_1",
    ]
    assert all("_source_scenario_id" not in scenario for scenario in scenarios)

    video_env.close()


def test_eval_throughput_uses_first_success_or_timeout_steps(
    tmp_path: Path,
) -> None:
    """Throughput should match the user's effective env-step definition."""
    runner = object.__new__(EmbodiedEvalRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "eval_throughput_path": str(tmp_path / "eval_throughput.json")
            }
        }
    )
    runner.logger = logging.getLogger(__name__)

    runner._save_eval_throughput(
        {
            "success_once": torch.tensor([True, True, False]),
            "first_success_step": torch.tensor([200, 350, -1]),
            "episode_len": torch.tensor([450, 450, 450]),
        }
    )

    throughput = json.loads(
        (tmp_path / "eval_throughput.json").read_text(encoding="utf-8")
    )
    assert throughput["num_success"] == 2
    assert throughput["successful_effective_steps"] == 550
    assert throughput["failed_effective_steps"] == 450
    assert throughput["total_effective_steps"] == 1000
    assert throughput["success_throughput_per_step"] == 0.002
    assert throughput["successes_per_1000_steps"] == 2.0


def test_isaaclab_metrics_retain_each_envs_first_success_step() -> None:
    """Later positive rewards must not overwrite the first success step."""
    env = object.__new__(IsaaclabBaseEnv)
    env.num_envs = 3
    env.device = torch.device("cpu")
    env.worker_info = object()
    env._episode_ids = torch.zeros(3, dtype=torch.int64)
    env._current_scenario_records = [None, None, None]
    env.init_cube_positions = None
    env.returns = torch.zeros(3)
    env.success_once = torch.zeros(3, dtype=torch.bool)
    env._first_success_steps = torch.full((3,), -1, dtype=torch.int64)

    env._elapsed_steps = torch.tensor([200, 349, 449], dtype=torch.int32)
    env._record_metrics(torch.tensor([1.0, 0.0, 0.0]), None, {})
    env._elapsed_steps += 1
    infos = env._record_metrics(torch.tensor([1.0, 1.0, 0.0]), None, {})

    assert infos["episode"]["first_success_step"].tolist() == [200, 350, -1]


def test_env_episode_json_supports_evaluations_without_scenarios(
    tmp_path: Path,
) -> None:
    """The env/episode naming mode must support the default non-scenario env."""
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "runner": {"policy_source": "/models/policy"},
            "rollout": {"model": {"model_path": "/models/policy"}},
            "env": {
                "eval": {
                    "per_episode_result_file_naming": "env_episode",
                }
            },
        }
    )
    worker._rank = 0
    worker.eval_env_list = [object()]
    worker.eval_per_episode_result_dir = tmp_path / "results"
    worker.eval_per_episode_skip_existing = False

    worker._write_eval_per_episode_results(
        {
            "scenario_id": torch.tensor([-1]),
            "success_once": torch.tensor([False]),
            "env_id": torch.tensor([0]),
            "episode_id": torch.tensor([0]),
            "worker_rank": torch.tensor([0]),
        },
        stage_id=0,
    )

    result_path = (
        tmp_path / "results" / "worker_000_stage_00_env_000_episode_000000.json"
    )
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["scenario_id"] == "-1"
    assert result["policy_success"] is False


def test_scenario_file_validation_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Explicit scenario files must be unambiguous to the loader."""
    record = _DummyVectorEnv().get_episode_scenario_record(0, 0)
    record.pop("_source_scenario_id")
    scenario_path = tmp_path / "scenarios.jsonl"
    scenario_path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    try:
        validate_scenario_file(scenario_path)
    except ValueError as exc:
        assert "duplicate id" in str(exc)
    else:
        raise AssertionError("duplicate scenario IDs should be rejected")
