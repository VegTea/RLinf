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

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np


CUBE_SCENARIO_PREFIXES = ("cube_1_", "cube_2_", "cube_3_")


@dataclass
class IsaacLabReplayTrajectory:
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        if "action" in self.arrays:
            return int(self.arrays["action"].shape[0])
        for value in self.arrays.values():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                return int(value.shape[0])
        return 0

    @property
    def label(self) -> str:
        return str(self.metadata.get("label", self.path.parent.name))

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class VisualScenarioSampler:
    scenario_file: Path
    records: list[dict[str, Any]]
    seed: int = 0

    @classmethod
    def from_jsonl(cls, scenario_file: str | os.PathLike, seed: int = 0):
        path = Path(scenario_file).expanduser().resolve()
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "id" not in record:
                    raise KeyError(f"Missing scenario id at {path}:{line_no}")
                records.append(record)
        if not records:
            raise ValueError(f"No visual scenario records found in {path}")
        return cls(scenario_file=path, records=records, seed=int(seed))

    def sample(self, trajectory_key: str, variant_idx: int) -> dict[str, Any]:
        rng = random.Random(f"{self.seed}:{trajectory_key}:{int(variant_idx)}")
        return dict(self.records[rng.randrange(len(self.records))])


def strip_cube_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith(CUBE_SCENARIO_PREFIXES)
    }


def discover_npz_files(path: str | os.PathLike, max_files: int | None = None) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("*.npz"))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .npz files found under {root}")
    return files


def load_replay_trajectory(path: str | os.PathLike) -> IsaacLabReplayTrajectory:
    npz_path = Path(path).expanduser().resolve()
    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.files:
            arrays[key] = np.asarray(data[key])
    metadata = {}
    metadata_value = arrays.get("metadata_json")
    if metadata_value is not None:
        try:
            metadata = json.loads(str(metadata_value.item()))
        except (json.JSONDecodeError, ValueError, AttributeError):
            metadata = {}
    return IsaacLabReplayTrajectory(path=npz_path, arrays=arrays, metadata=metadata)


def frame_payload(traj: IsaacLabReplayTrajectory, frame_idx: int, env_id: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {"env_id": int(env_id)}
    for key in ("joint_pos", "joint_vel", "object", "cube_positions", "cube_orientations"):
        if key in traj.arrays:
            payload[key] = traj.arrays[key][frame_idx].astype(np.float32)
    return payload


def initial_payload(traj: IsaacLabReplayTrajectory, env_id: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {"env_id": int(env_id)}
    key_map = {
        "initial_joint_pos": "joint_pos",
        "initial_joint_vel": "joint_vel",
        "initial_object": "object",
        "initial_cube_positions": "cube_positions",
        "initial_cube_orientations": "cube_orientations",
    }
    for src_key, dst_key in key_map.items():
        if src_key in traj.arrays:
            payload[dst_key] = traj.arrays[src_key].astype(np.float32)
    return payload


def image_from_obs_key(obs: dict[str, Any], key: str) -> np.ndarray:
    policy_obs = obs.get("policy", obs) if isinstance(obs, dict) else {}
    image = np.asarray(policy_obs[key])
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def depth_from_obs_key(obs: dict[str, Any], key: str) -> np.ndarray:
    policy_obs = obs.get("policy", obs) if isinstance(obs, dict) else {}
    depth = np.asarray(policy_obs[key])
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    return depth.astype(np.float32, copy=False)


def table_image_from_obs(obs: dict[str, Any], camera: str = "table") -> np.ndarray:
    key = "table_cam" if camera == "table" else "wrist_cam"
    return image_from_obs_key(obs, key)


def depth_from_obs(obs: dict[str, Any], camera: str = "table") -> np.ndarray:
    key = "table_cam_depth" if camera == "table" else "wrist_cam_depth"
    return depth_from_obs_key(obs, key)


def detect_reset_like_tail(
    traj: IsaacLabReplayTrajectory,
    distance_threshold: float = 0.25,
) -> dict[str, Any]:
    if traj.length < 2 or "object" not in traj.arrays:
        return {"suspect_reset_frame": False, "reason": "missing_object_or_short"}
    object_states = traj.arrays["object"]
    prev_cube_pos = object_states[-2, :21].reshape(3, 7)[:, :3]
    last_cube_pos = object_states[-1, :21].reshape(3, 7)[:, :3]
    max_jump = float(np.linalg.norm(last_cube_pos - prev_cube_pos, axis=1).max())
    return {
        "suspect_reset_frame": max_jump > float(distance_threshold),
        "max_cube_position_jump": max_jump,
        "distance_threshold": float(distance_threshold),
    }


def write_video(frames: list[np.ndarray], path: str | os.PathLike, fps: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=int(fps)) as writer:
        for frame in frames:
            writer.append_data(frame)


def write_manifest_record(path: str | os.PathLike, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True) + "\n")
