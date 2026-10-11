# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Thin reader for the pinned Stack-Cube LeRobot v2.1 parquet/video layout."""

import json
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from rlinf.models.embodiment.dem.stack_cube_io import INSTRUCTION, rotation_audit


class StackCubeDataset(Dataset):
    """Current dual-camera observation and H actions; never pad/cross episodes."""

    def __init__(self, root: str, episodes: list[int], horizon: int = 16):
        self.root = Path(root)
        self.horizon = horizon
        if horizon <= 0 or not episodes or len(set(episodes)) != len(episodes):
            raise ValueError("Require positive horizon and nonempty unique episode IDs")
        info = json.loads((self.root / "meta/info.json").read_text())
        if info["fps"] != 20 or info["codebase_version"] != "v2.1":
            raise ValueError("Expected the 20 Hz LeRobot v2.1 Stack-Cube dataset")
        self.states, self.actions, self.timestamps = {}, {}, {}
        self.indices = []
        for episode in episodes:
            path = self.root / f"data/chunk-000/episode_{episode:06d}.parquet"
            rows = pq.read_table(path).to_pydict()
            states = np.asarray(rows["observation.state"], dtype=np.float32)
            actions = np.asarray(rows["action"], dtype=np.float32)
            count = len(states)
            if states.shape != (count, 8) or actions.shape != (count, 7):
                raise ValueError(f"Bad state/action shape in {path}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"Non-finite data in {path}")
            if rows["frame_index"] != list(range(count)) or set(
                rows["episode_index"]
            ) != {episode}:
                raise ValueError(f"Bad frame/episode indices in {path}")
            times = np.asarray(rows["timestamp"])
            if not np.allclose(times, np.arange(count) / 20, atol=1e-4):
                raise ValueError(f"Nonuniform/misaligned timestamps in {path}")
            if count < horizon:
                raise ValueError(f"Episode {episode} shorter than horizon")
            self.states[episode], self.actions[episode] = states, actions
            self.timestamps[episode] = times
            self.indices.extend(
                (episode, frame) for frame in range(count - horizon + 1)
            )

    # ponytail: 8 decoded camera videos (~600MB) per process; disk-backed cache
    # only if random full-dataset access makes video decoding the bottleneck.
    @lru_cache(maxsize=8)
    def frames(self, episode: int, camera: str) -> np.ndarray:
        """Decode RGB and verify every video timestamp against its parquet row."""
        path = (
            self.root
            / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
        )
        images, timestamps = [], []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                images.append(frame.to_ndarray(format="rgb24"))
                timestamps.append(float(frame.pts * frame.time_base))
        expected = self.timestamps[episode]
        if len(images) != len(expected) or not np.allclose(
            timestamps, expected, atol=1e-4
        ):
            raise ValueError(f"Video/parquet timestamps do not align: {path}")
        result = np.stack(images)
        if result.shape[1:] != (256, 256, 3):
            raise ValueError(f"Bad camera shape: {path}: {result.shape}")
        return result

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        episode, frame = self.indices[index]
        return {
            "images": torch.from_numpy(
                np.stack(
                    [self.frames(episode, key)[frame] for key in ("front", "wrist")]
                )
            ),
            "proprio": torch.from_numpy(self.states[episode][frame].copy()),
            "actions": torch.from_numpy(
                self.actions[episode][frame : frame + self.horizon].copy()
            ),
            "instruction": INSTRUCTION,
            "episode_id": episode,
            "frame_index": frame,
        }

    def statistics(self) -> dict[str, torch.Tensor]:
        """Count original frames once, including tail frames, not overlapping chunks."""
        result = {}
        for name, arrays in (("state", self.states), ("action", self.actions)):
            values = torch.from_numpy(np.concatenate(list(arrays.values())))
            result[f"{name}_mean"] = values.mean(0)
            result[f"{name}_std"] = values.std(0).clamp_min(1e-4)
        return result

    def audit(self, check_videos: bool = True) -> dict:
        """Report evidence without claiming acquisition provenance was established."""
        per_episode = {}
        for episode, states in self.states.items():
            if check_videos:
                for key in ("front", "wrist"):
                    self.frames(episode, key)
            per_episode[str(episode)] = rotation_audit(states)
        actions = np.concatenate(list(self.actions.values()))
        states = np.concatenate(list(self.states.values()))
        return {
            "episodes": len(self.states),
            "frames": len(states),
            "samples": len(self),
            "horizon": self.horizon,
            "video_timestamps_checked": check_videos,
            "action_min": actions.min(0).tolist(),
            "action_max": actions.max(0).tolist(),
            "state_min": states.min(0).tolist(),
            "state_max": states.max(0).tolist(),
            "gripper_commands": np.unique(actions[:, -1]).tolist(),
            "rotation_hypotheses": per_episode,
            "contract_verified": False,
            "remaining": "Confirm acquisition rotation order, action units and observation/action alignment against simulator; continuity alone is not proof.",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    info = json.loads((Path(args.data) / "meta/info.json").read_text())
    dataset = StackCubeDataset(args.data, list(range(info["total_episodes"])))
    report = dataset.audit()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
