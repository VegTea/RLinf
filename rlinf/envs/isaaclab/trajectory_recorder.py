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
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux cluster path uses fcntl.
    fcntl = None


class IsaacLabTrajectoryRecorder:
    """Save lightweight per-episode eval trajectories without image tensors."""

    LOW_DIM_POLICY_KEYS = (
        "joint_pos",
        "joint_vel",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
        "cube_positions",
        "cube_orientations",
        "object",
        "actions",
    )

    def __init__(self, cfg, num_envs: int, worker_rank: int, seed: int):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.num_envs = int(num_envs)
        self.worker_rank = int(worker_rank)
        self.seed = int(seed)
        self.output_dir = os.path.abspath(str(getattr(cfg, "output_dir", "")))
        self.target_success = int(getattr(cfg, "target_success", 1000))
        self.target_fail = int(getattr(cfg, "target_fail", 1000))
        self.record_success = bool(getattr(cfg, "record_success", True))
        self.record_fail = bool(getattr(cfg, "record_fail", True))
        self.flush_manifest = bool(getattr(cfg, "flush_manifest", True))

        self.state_file = os.path.join(
            self.output_dir,
            str(getattr(cfg, "state_file", "quota_state.json")),
        )
        self.manifest_file = os.path.join(
            self.output_dir,
            str(getattr(cfg, "manifest_file", "manifest.jsonl")),
        )
        self.lock_file = self.state_file + ".lock"

        self._buffers = [defaultdict(list) for _ in range(self.num_envs)]
        self._initial_obs: list[dict[str, np.ndarray]] = [
            {} for _ in range(self.num_envs)
        ]
        self._scenario_records: list[dict[str, Any] | None] = [None] * self.num_envs
        self._episode_closed = [False] * self.num_envs

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "success"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "fail"), exist_ok=True)
            self._ensure_state_file()

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        scenario_records: dict[int, dict[str, Any]] | None = None,
        raw_obs: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        for env_id in self._iter_env_ids(env_ids):
            self._buffers[env_id] = defaultdict(list)
            self._initial_obs[env_id] = self._extract_initial_obs(raw_obs, env_id)
            self._episode_closed[env_id] = False
            if scenario_records is not None:
                record = scenario_records.get(env_id)
                self._scenario_records[env_id] = dict(record) if record else None

    def record_step(
        self,
        actions: torch.Tensor | np.ndarray | None,
        raw_obs: dict[str, Any],
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        dones: torch.Tensor,
        success_hits: torch.Tensor,
        infos: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        policy_obs = raw_obs.get("policy", {}) if isinstance(raw_obs, dict) else {}
        episode_info = infos.get("episode", {}) if isinstance(infos, dict) else {}

        for env_id in range(self.num_envs):
            if self._episode_closed[env_id]:
                continue

            buffer = self._buffers[env_id]
            self._append_value(buffer, "action", actions, env_id)
            self._append_value(buffer, "reward", rewards, env_id)
            self._append_value(buffer, "termination", terminations, env_id)
            self._append_value(buffer, "truncation", truncations, env_id)

            for key in self.LOW_DIM_POLICY_KEYS:
                if key in policy_obs:
                    output_key = "obs_action" if key == "actions" else key
                    self._append_value(buffer, output_key, policy_obs[key], env_id)

            if bool(self._value_for_env(success_hits, env_id)):
                self._flush_episode(
                    env_id,
                    episode_info,
                    terminations,
                    truncations,
                    label_override="success",
                    stop_reason="success_hit",
                )
                self._episode_closed[env_id] = True
            elif bool(self._value_for_env(dones, env_id)):
                self._flush_episode(
                    env_id,
                    episode_info,
                    terminations,
                    truncations,
                    label_override="fail",
                    stop_reason="episode_done",
                )
                self._episode_closed[env_id] = True

    def get_counts(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "success": 0, "fail": 0}
        return {"enabled": True, **self._read_state()}

    def targets_met(self) -> bool:
        counts = self.get_counts()
        return (
            int(counts.get("success", 0)) >= self.target_success
            and int(counts.get("fail", 0)) >= self.target_fail
        )

    def _append_value(self, buffer: dict[str, list], key: str, value: Any, env_id: int):
        item = self._value_for_env(value, env_id)
        if item is None:
            return
        buffer[key].append(np.asarray(item))

    def _flush_episode(
        self,
        env_id: int,
        episode_info: dict[str, Any],
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        label_override: str | None = None,
        stop_reason: str = "episode_done",
    ) -> None:
        buffer = self._buffers[env_id]
        if not buffer:
            return

        success_once = bool(
            self._value_for_env(episode_info.get("success_once"), env_id)
        )
        label = label_override or ("success" if success_once else "fail")
        if label == "success" and not self.record_success:
            return
        if label == "fail" and not self.record_fail:
            return

        quota_target = self.target_success if label == "success" else self.target_fail
        quota = self._reserve_quota(label, quota_target)
        if quota is None:
            return

        episode_id = int(
            self._value_for_env(episode_info.get("episode_id"), env_id) or 0
        )
        file_name = (
            f"{label}_{quota:06d}_rank{self.worker_rank:03d}_"
            f"env{env_id:03d}_episode{episode_id:06d}.npz"
        )
        rel_path = os.path.join(label, file_name)
        abs_path = os.path.join(self.output_dir, rel_path)

        arrays = {}
        for key, value in self._initial_obs[env_id].items():
            arrays[f"initial_{key}"] = value
        for key, values in buffer.items():
            if not values:
                continue
            arrays[key] = np.stack(values, axis=0)

        success_at_end = bool(self._value_for_env(terminations, env_id))
        truncated = bool(self._value_for_env(truncations, env_id))
        metadata = {
            "path": rel_path,
            "label": label,
            "quota_index": quota,
            "worker_rank": self.worker_rank,
            "env_id": env_id,
            "episode_id": episode_id,
            "seed": self.seed,
            "length": len(next(iter(buffer.values()))) if buffer else 0,
            "success_once": success_once,
            "success_at_end": success_at_end,
            "truncated": truncated,
            "stop_reason": stop_reason,
            "scenario": self._scenario_records[env_id],
        }

        arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=True))
        np.savez_compressed(abs_path, **arrays)
        self._append_manifest(metadata)

    def _extract_initial_obs(self, raw_obs: dict[str, Any] | None, env_id: int):
        if not isinstance(raw_obs, dict):
            return {}
        policy_obs = raw_obs.get("policy", {})
        if not isinstance(policy_obs, dict):
            return {}

        initial = {}
        for key in self.LOW_DIM_POLICY_KEYS:
            if key not in policy_obs:
                continue
            output_key = "obs_action" if key == "actions" else key
            value = self._value_for_env(policy_obs[key], env_id)
            if value is not None:
                initial[output_key] = np.asarray(value)
        return initial

    def _reserve_quota(self, label: str, target: int) -> int | None:
        with self._locked_state() as state:
            current = int(state.get(label, 0))
            if current >= target:
                state[f"skipped_{label}"] = int(state.get(f"skipped_{label}", 0)) + 1
                return None
            state[label] = current + 1
            return current

    def _append_manifest(self, metadata: dict[str, Any]) -> None:
        with open(self.lock_file, "a+", encoding="utf-8") as lock_fp:
            if fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                with open(self.manifest_file, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(metadata, ensure_ascii=True) + "\n")
                    if self.flush_manifest:
                        fp.flush()
                        os.fsync(fp.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def _ensure_state_file(self) -> None:
        with self._locked_state() as state:
            state.setdefault("success", self._count_existing("success"))
            state.setdefault("fail", self._count_existing("fail"))
            state.setdefault("skipped_success", 0)
            state.setdefault("skipped_fail", 0)

    @contextmanager
    def _locked_state(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.lock_file, "a+", encoding="utf-8") as lock_fp:
            if fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state()
                yield state
                tmp_path = self.state_file + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as state_fp:
                    json.dump(
                        state,
                        state_fp,
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    )
                    state_fp.write("\n")
                    state_fp.flush()
                    os.fsync(state_fp.fileno())
                os.replace(tmp_path, self.state_file)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def _read_state(self) -> dict[str, int]:
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except json.JSONDecodeError:
            return {}

    def _count_existing(self, label: str) -> int:
        label_dir = os.path.join(self.output_dir, label)
        if not os.path.isdir(label_dir):
            return 0
        return sum(1 for name in os.listdir(label_dir) if name.endswith(".npz"))

    def _iter_env_ids(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            yield from range(self.num_envs)
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        for env_id in env_ids:
            yield int(env_id)

    def _value_for_env(self, value: Any, env_id: int):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.shape == ():
                return value.detach().cpu().item()
            return value[env_id].detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value[env_id]
        if isinstance(value, (list, tuple)):
            return value[env_id]
        return value
