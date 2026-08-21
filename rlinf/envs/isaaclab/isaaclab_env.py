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

import copy
import json
import os
from typing import Optional

import gymnasium as gym
import imageio
import numpy as np
import torch
from omegaconf import open_dict

from rlinf.envs.isaaclab.trajectory_recorder import IsaacLabTrajectoryRecorder
from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv


class IsaaclabBaseEnv(gym.Env):
    """
    Class for isaaclab in rlinf. Different from other lab enviromnent, the output of isaaclab is all tensor on
    cuda.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        self.cfg = cfg
        self.isaaclab_env_id = self.cfg.init_params.id
        self.num_envs = num_envs

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.video_cfg = cfg.video_cfg
        self._init_isaaclab_env()
        self.device = self.env.device()

        self.task_description = cfg.init_params.task_description
        self._is_start = True  # if this is first time for simulator
        self.auto_reset = cfg.auto_reset
        self.prev_step_reward = torch.zeros(self.num_envs).to(self.device)
        self.use_rel_reward = cfg.use_rel_reward
        self.init_cube_positions = None
        self._reset_snapshot_counter = 0
        self._final_snapshot_counter = 0
        self._episode_ids = torch.zeros(self.num_envs, dtype=torch.int64).to(
            self.device
        )
        self._has_reset_once = torch.zeros(self.num_envs, dtype=torch.bool).to(
            self.device
        )
        self._current_scenario_records: list[dict | None] = [None] * self.num_envs

        self._init_metrics()
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32).to(
            self.device
        )
        self.ignore_terminations = cfg.ignore_terminations
        trajectory_cfg = cfg.get("trajectory_record_cfg", None)
        self.trajectory_recorder = IsaacLabTrajectoryRecorder(
            trajectory_cfg,
            num_envs=self.num_envs,
            worker_rank=getattr(self.worker_info, "rank", 0),
            seed=self.seed,
        ) if trajectory_cfg is not None else None

    def _make_env_function(self):
        raise NotImplementedError

    def _init_isaaclab_env(self):
        env_fn = self._make_env_function()
        self.env = SubProcIsaacLabEnv(env_fn)
        self.env.reset(seed=self.seed)

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.fail_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.returns = torch.zeros(self.num_envs).to(self.device)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool).to(self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | (step_reward > 0)
        # batch level
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["env_id"] = torch.arange(self.num_envs, device=self.device)
        episode_info["episode_id"] = self._episode_ids.clone()
        episode_info["worker_rank"] = torch.full(
            (self.num_envs,),
            int(getattr(self.worker_info, "rank", 0)),
            dtype=torch.int64,
            device=self.device,
        )
        scenario_ids = []
        for record in self._current_scenario_records:
            try:
                scenario_ids.append(int(record.get("id", -1)) if record else -1)
            except (TypeError, ValueError):
                scenario_ids.append(-1)
        episode_info["scenario_id"] = torch.tensor(
            scenario_ids, dtype=torch.int64, device=self.device
        )
        if self.init_cube_positions is not None:
            episode_info["init_cube_positions"] = self.init_cube_positions.clone()
            episode_info["init_cube_1_pos"] = self.init_cube_positions[:, 0].clone()
            episode_info["init_cube_2_pos"] = self.init_cube_positions[:, 1].clone()
            episode_info["init_cube_3_pos"] = self.init_cube_positions[:, 2].clone()
        infos["episode"] = episode_info
        return infos

    def _extract_init_cube_positions(self, raw_obs):
        if not isinstance(raw_obs, dict) or "policy" not in raw_obs:
            return None

        policy_obs = raw_obs["policy"]
        object_obs = policy_obs.get("object", None)
        if object_obs is not None and object_obs.shape[-1] >= 17:
            return torch.stack(
                (
                    object_obs[:, 0:3],
                    object_obs[:, 7:10],
                    object_obs[:, 14:17],
                ),
                dim=1,
            )

        cube_positions = policy_obs.get("cube_positions", None)
        if cube_positions is not None and cube_positions.shape[-1] >= 9:
            return cube_positions.reshape(cube_positions.shape[0], 3, 3)

        return None

    def _record_init_cube_positions(self, raw_obs, env_ids=None):
        init_cube_positions = self._extract_init_cube_positions(raw_obs)
        if init_cube_positions is None:
            if self.cfg.get("debug_init_cube_positions", False):
                print(
                    "[IsaacLab debug] failed to extract init cube positions from reset obs",
                    flush=True,
                )
            return

        init_cube_positions = init_cube_positions.clone()
        if self.init_cube_positions is None:
            self.init_cube_positions = torch.zeros(
                self.num_envs,
                3,
                3,
                dtype=init_cube_positions.dtype,
                device=init_cube_positions.device,
            )

        if env_ids is None or init_cube_positions.shape[0] == self.num_envs:
            self.init_cube_positions[:] = init_cube_positions[: self.num_envs]
        else:
            self.init_cube_positions[env_ids] = init_cube_positions

        if self.cfg.get("debug_init_cube_positions", False) and not getattr(
            self, "_printed_init_cube_positions", False
        ):
            sample = self.init_cube_positions[:2].detach().cpu().tolist()
            print(
                "[IsaacLab debug] init_cube_positions shape: "
                f"{tuple(self.init_cube_positions.shape)}, sample: {sample}",
                flush=True,
            )
            self._printed_init_cube_positions = True

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[torch.Tensor] = None,
    ):
        if env_ids is None:
            target_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            target_env_ids = env_ids.to(self.device)

        if target_env_ids.numel() > 0:
            seen_mask = self._has_reset_once[target_env_ids]
            if seen_mask.any():
                self._episode_ids[target_env_ids[seen_mask]] += 1

        if env_ids is None:
            obs, reset_info = self.env.reset(seed=seed)
        else:
            obs, reset_info = self.env.reset(seed=seed, env_ids=env_ids)
        infos = {}
        scenario_records = self._extract_scenario_records(reset_info)
        self._record_scenario_records(scenario_records)
        self._record_init_cube_positions(obs, env_ids)
        self._save_reset_snapshots(obs, env_ids)
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.reset(target_env_ids, scenario_records, obs)
        if target_env_ids.numel() > 0:
            self._has_reset_once[target_env_ids] = True
        obs = self._wrap_obs(obs)
        self._reset_metrics(env_ids)
        return obs, infos

    def _extract_scenario_records(self, reset_info):
        if not isinstance(reset_info, dict):
            return {}
        records = reset_info.get("scenario_records", {})
        if not isinstance(records, dict):
            return {}
        return {int(env_id): record for env_id, record in records.items()}

    def _record_scenario_records(self, scenario_records):
        for env_id, record in scenario_records.items():
            if 0 <= env_id < self.num_envs:
                self._current_scenario_records[env_id] = record

    def _to_numpy_image_batch(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            image_tensor = image_tensor.detach().cpu().numpy()
        else:
            image_tensor = np.asarray(image_tensor)

        if image_tensor.ndim == 3:
            image_tensor = image_tensor[None]

        if image_tensor.ndim != 4:
            return None

        if image_tensor.shape[1] in (1, 3, 4) and image_tensor.shape[-1] not in (
            1,
            3,
            4,
        ):
            image_tensor = np.transpose(image_tensor, (0, 2, 3, 1))

        if image_tensor.dtype != np.uint8:
            image_tensor = np.clip(image_tensor, 0, 255).astype(np.uint8)

        return image_tensor

    def _get_reset_snapshot_root(self):
        snapshot_cfg = getattr(self.cfg, "reset_snapshot_cfg", None)
        if snapshot_cfg is None or not getattr(snapshot_cfg, "enabled", False):
            return None

        root_dir = snapshot_cfg.output_dir
        worker_rank = getattr(self.worker_info, "rank", 0)
        return os.path.join(root_dir, f"worker_{worker_rank:03d}")

    def _save_single_reset_snapshot(
        self,
        image,
        image_kind,
        root_dir,
        reset_id,
        env_id,
        cube_pos,
    ):
        image_dir = os.path.join(root_dir, image_kind)
        os.makedirs(image_dir, exist_ok=True)
        x = float(cube_pos[0].item())
        y = float(cube_pos[1].item())
        file_name = (
            f"reset_{reset_id:06d}_env{env_id:03d}_"
            f"x{x:+.4f}_y{y:+.4f}.png"
        )
        imageio.imwrite(os.path.join(image_dir, file_name), image)
        return file_name

    def _save_reset_snapshots(self, raw_obs, env_ids=None):
        root_dir = self._get_reset_snapshot_root()
        if root_dir is None or self.init_cube_positions is None:
            return

        snapshot_cfg = self.cfg.reset_snapshot_cfg
        policy_obs = raw_obs.get("policy", {}) if isinstance(raw_obs, dict) else {}
        table_images = self._to_numpy_image_batch(policy_obs.get("table_cam"))
        wrist_images = self._to_numpy_image_batch(policy_obs.get("wrist_cam"))

        if env_ids is None:
            local_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            local_env_ids = env_ids.to(self.device)

        if table_images is None and wrist_images is None:
            return

        os.makedirs(root_dir, exist_ok=True)
        manifest_path = os.path.join(root_dir, "manifest.jsonl")

        with open(manifest_path, "a", encoding="utf-8") as manifest_fp:
            for batch_idx, env_id_tensor in enumerate(local_env_ids):
                env_id = int(env_id_tensor.item())
                reset_id = self._reset_snapshot_counter
                self._reset_snapshot_counter += 1

                cube_2_pos = self.init_cube_positions[env_id, 1].detach().cpu()
                record = {
                    "reset_id": reset_id,
                    "env_id": env_id,
                    "episode_id": int(self._episode_ids[env_id].item()),
                    "worker_rank": getattr(self.worker_info, "rank", 0),
                    "seed": int(self.seed),
                    "cube_2_xyz": [float(v) for v in cube_2_pos.tolist()],
                }

                if (
                    getattr(snapshot_cfg, "save_table_png", True)
                    and table_images is not None
                    and batch_idx < table_images.shape[0]
                ):
                    record["table_png"] = self._save_single_reset_snapshot(
                        table_images[batch_idx],
                        "table",
                        root_dir,
                        reset_id,
                        env_id,
                        cube_2_pos,
                    )

                if (
                    getattr(snapshot_cfg, "save_wrist_png", False)
                    and wrist_images is not None
                    and batch_idx < wrist_images.shape[0]
                ):
                    record["wrist_png"] = self._save_single_reset_snapshot(
                        wrist_images[batch_idx],
                        "wrist",
                        root_dir,
                        reset_id,
                        env_id,
                        cube_2_pos,
                    )

                if getattr(snapshot_cfg, "save_metadata", True):
                    manifest_fp.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _get_final_snapshot_root(self):
        snapshot_cfg = getattr(self.cfg, "final_snapshot_cfg", None)
        if snapshot_cfg is None or not getattr(snapshot_cfg, "enabled", False):
            return None

        root_dir = snapshot_cfg.output_dir
        worker_rank = getattr(self.worker_info, "rank", 0)
        return os.path.join(root_dir, f"worker_{worker_rank:03d}")

    def _tensor_value_for_env(self, value, env_id):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.shape == ():
                return value.detach().cpu().item()
            return value[env_id].detach().cpu()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value[env_id]
        if isinstance(value, (list, tuple)):
            return value[env_id]
        return value

    def _save_single_final_snapshot(
        self,
        image,
        image_kind,
        root_dir,
        snapshot_id,
        label,
        env_id,
        episode_id,
    ):
        image_dir = os.path.join(root_dir, label, image_kind)
        os.makedirs(image_dir, exist_ok=True)
        file_name = (
            f"success_hit_{snapshot_id:06d}_env{env_id:03d}_"
            f"episode{episode_id:06d}_{label}.png"
        )
        imageio.imwrite(os.path.join(image_dir, file_name), image)
        return os.path.join(label, image_kind, file_name)

    def _save_success_hit_snapshots(
        self,
        raw_obs,
        success_hits,
        infos,
        terminations,
        truncations,
    ):
        root_dir = self._get_final_snapshot_root()
        if root_dir is None or not success_hits.any():
            return

        snapshot_cfg = self.cfg.final_snapshot_cfg
        policy_obs = raw_obs.get("policy", {}) if isinstance(raw_obs, dict) else {}
        table_images = self._to_numpy_image_batch(policy_obs.get("table_cam"))
        wrist_images = self._to_numpy_image_batch(policy_obs.get("wrist_cam"))
        if table_images is None and wrist_images is None:
            return

        episode_info = infos.get("episode", {}) if isinstance(infos, dict) else {}
        final_cube_positions = self._extract_init_cube_positions(raw_obs)
        hit_env_ids = torch.arange(self.num_envs, device=self.device)[success_hits]

        os.makedirs(root_dir, exist_ok=True)
        manifest_path = os.path.join(root_dir, "manifest.jsonl")
        with open(manifest_path, "a", encoding="utf-8") as manifest_fp:
            for env_id_tensor in hit_env_ids:
                env_id = int(env_id_tensor.item())
                snapshot_id = self._final_snapshot_counter
                self._final_snapshot_counter += 1

                label = "success_hit"
                episode_id = int(
                    self._tensor_value_for_env(
                        episode_info.get("episode_id"), env_id
                    )
                    or 0
                )
                cube_positions = None
                if final_cube_positions is not None:
                    cube_positions = final_cube_positions[env_id].detach().cpu()
                elif self.init_cube_positions is not None:
                    cube_positions = self.init_cube_positions[env_id].detach().cpu()

                record = {
                    "snapshot_id": snapshot_id,
                    "label": label,
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "worker_rank": getattr(self.worker_info, "rank", 0),
                    "seed": int(self.seed),
                    "trigger": "success_hit",
                    "success_once": True,
                    "success_at_end": bool(
                        self._tensor_value_for_env(terminations, env_id)
                    ),
                    "truncated": bool(
                        self._tensor_value_for_env(truncations, env_id)
                    ),
                }
                if cube_positions is not None:
                    record["cube_xyz"] = [
                        [float(v) for v in cube_pos.tolist()]
                        for cube_pos in cube_positions
                    ]

                if (
                    getattr(snapshot_cfg, "save_table_png", True)
                    and table_images is not None
                    and env_id < table_images.shape[0]
                ):
                    record["table_png"] = self._save_single_final_snapshot(
                        table_images[env_id],
                        "table",
                        root_dir,
                        snapshot_id,
                        label,
                        env_id,
                        episode_id,
                    )

                if (
                    getattr(snapshot_cfg, "save_wrist_png", False)
                    and wrist_images is not None
                    and env_id < wrist_images.shape[0]
                ):
                    record["wrist_png"] = self._save_single_final_snapshot(
                        wrist_images[env_id],
                        "wrist",
                        root_dir,
                        snapshot_id,
                        label,
                        env_id,
                        episode_id,
                    )

                if getattr(snapshot_cfg, "save_metadata", True):
                    manifest_fp.write(json.dumps(record, ensure_ascii=True) + "\n")

    def step(self, actions=None, auto_reset=True):
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)

        step_reward = step_reward.clone()
        terminations = terminations.clone()
        truncations = truncations.clone()
        success_hits = (~self.success_once) & (step_reward > 0)

        self._elapsed_steps += 1

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations

        dones = terminations | truncations

        infos = self._record_metrics(
            step_reward, terminations, {}
        )  # return infos is useless
        self._save_success_hit_snapshots(
            obs, success_hits, infos, terminations, truncations
        )
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.record_step(
                actions=actions,
                raw_obs=obs,
                rewards=step_reward,
                terminations=terminations,
                truncations=truncations,
                dones=dones,
                success_hits=success_hits,
                infos=infos,
            )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False

        obs = self._wrap_obs(obs)

        _auto_reset = auto_reset and self.auto_reset  # always False
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            step_reward,
            terminations,
            truncations,
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations).to(
                self.device
            )
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations).to(self.device)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = torch.arange(0, self.num_envs).to(dones.device)
        env_idx = env_idx[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_ids=env_idx,
        )

        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _wrap_obs(self, obs):
        raise NotImplementedError

    def close(self):
        self.env.close()

    def update_reset_state_ids(self):
        """
        No muti task.
        """
        pass

    def set_scenario_curriculum_stage(self, stage_index):
        if hasattr(self.env, "set_scenario_curriculum_stage"):
            return self.env.set_scenario_curriculum_stage(stage_index)
        return {"enabled": False, "reason": "scenario_curriculum_not_supported"}

    def set_scenario_curriculum_progress(self, stage_index=None, stage_step=None):
        if hasattr(self.env, "set_scenario_curriculum_progress"):
            return self.env.set_scenario_curriculum_progress(stage_index, stage_step)
        return {"enabled": False, "reason": "scenario_curriculum_not_supported"}

    def get_scenario_curriculum_state(self):
        if hasattr(self.env, "get_scenario_curriculum_state"):
            return self.env.get_scenario_curriculum_state()
        return {"enabled": False, "reason": "scenario_curriculum_not_supported"}

    def get_trajectory_record_counts(self):
        if self.trajectory_recorder is None:
            return {"enabled": False, "success": 0, "fail": 0}
        return self.trajectory_recorder.get_counts()

    """
    Below codes are all copied from libero, thanks to the author of libero!
    """

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps.to(self.device)

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward
