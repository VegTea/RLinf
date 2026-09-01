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

from copy import deepcopy

import gymnasium as gym
import torch
from omegaconf import OmegaConf

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


class IsaaclabStackCubeEnv(IsaaclabBaseEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        """
        function for make isaaclab
        """

        def make_env_isaaclab():
            import os

            # Remove DISPLAY variable to force headless mode and avoid GLX errors
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            from isaaclab_tasks.utils import load_cfg_from_registry

            from rlinf.envs.isaaclab.rewarded_stack_cfg import (
                register_rewarded_stack_env,
            )

            register_rewarded_stack_env()

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            # Isaac Lab's stock Stack Cube config uses Nucleus URIs.  For the
            # split installation, point the robot and block assets at the local
            # Isaac Sim asset root so a complete Nucleus checkout is unnecessary.
            local_asset_root = os.environ.get("ISAACSIM_ASSET_ROOT")
            if local_asset_root and not local_asset_root.startswith(
                ("omniverse://", "http://", "https://")
            ):
                from pathlib import Path

                local_asset_root = Path(local_asset_root)
                # Asset ZIPs use ``Assets/Isaac/6.0/Isaac/...`` while some
                # extracted installations expose the contents directly under
                # ``Assets/Isaac/6.0``. Probe both layouts per asset.

                def _asset_path(*relative_paths):
                    candidates = [local_asset_root / rel for rel in relative_paths]
                    for candidate in candidates:
                        if candidate.is_file():
                            return candidate
                    return candidates[0]

                robot_usd = _asset_path(
                    "IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
                    "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
                )
                block_paths = {
                    "cube_1": _asset_path(
                        "Isaac/Props/Blocks/blue_block.usd",
                        "Props/Blocks/blue_block.usd",
                    ),
                    "cube_2": _asset_path(
                        "Isaac/Props/Blocks/red_block.usd",
                        "Props/Blocks/red_block.usd",
                    ),
                    "cube_3": _asset_path(
                        "Isaac/Props/Blocks/green_block.usd",
                        "Props/Blocks/green_block.usd",
                    ),
                }
                missing_assets = [
                    str(path)
                    for path in [robot_usd, *block_paths.values()]
                    if not path.is_file()
                ]
                if missing_assets:
                    raise FileNotFoundError(
                        "Local Isaac Sim asset root is missing Stack Cube assets: "
                        + ", ".join(missing_assets)
                    )
                isaac_env_cfg.scene.robot.spawn.usd_path = str(robot_usd)
                for cube_name, cube_path in block_paths.items():
                    getattr(isaac_env_cfg.scene, cube_name).spawn.usd_path = str(
                        cube_path
                    )
            # Seed the IsaacLab env config before construction so the simulator's
            # initial reset path is deterministic and doesn't warn about an unset seed.
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = (
                self.cfg.init_params.num_envs
            )  # default 4096 ant_env_spaces.pkl
            isaaclab_episode_length_steps = getattr(
                self.cfg.init_params, "isaaclab_episode_length_steps", None
            )
            if isaaclab_episode_length_steps is not None:
                isaaclab_episode_length_steps = int(isaaclab_episode_length_steps)
                if isaaclab_episode_length_steps <= 0:
                    raise ValueError(
                        "init_params.isaaclab_episode_length_steps must be positive, "
                        f"got {isaaclab_episode_length_steps}"
                    )
                control_dt = float(isaac_env_cfg.sim.dt) * int(isaac_env_cfg.decimation)
                isaac_env_cfg.episode_length_s = (
                    isaaclab_episode_length_steps * control_dt
                )
                print(
                    "[RLinf] Overriding IsaacLab episode_length_s to "
                    f"{isaac_env_cfg.episode_length_s:.6f}s "
                    f"({isaaclab_episode_length_steps} control steps)"
                )

            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height
            isaac_env_cfg.scene.wrist_cam.width = self.cfg.init_params.wrist_cam.width
            isaac_env_cfg.scene.table_cam.height = self.cfg.init_params.table_cam.height
            isaac_env_cfg.scene.table_cam.width = self.cfg.init_params.table_cam.width
            antialiasing_mode = self.cfg.init_params.get(
                "antialiasing_mode", None
            )
            if antialiasing_mode is not None:
                isaac_env_cfg.sim.render.antialiasing_mode = str(antialiasing_mode)

            if self.enable_depth_observations:
                from isaaclab.managers import ObservationTermCfg as ObsTerm
                from isaaclab.managers import SceneEntityCfg
                from isaaclab_tasks.manager_based.manipulation.stack import mdp

                isaac_env_cfg.observations.policy.table_cam_depth = ObsTerm(
                    func=mdp.image,
                    params={
                        "sensor_cfg": SceneEntityCfg("table_cam"),
                        "data_type": "distance_to_image_plane",
                        "normalize": False,
                    },
                )
                isaac_env_cfg.observations.policy.wrist_cam_depth = ObsTerm(
                    func=mdp.image,
                    params={
                        "sensor_cfg": SceneEntityCfg("wrist_cam"),
                        "data_type": "distance_to_image_plane",
                        "normalize": False,
                    },
                )
            else:
                isaac_env_cfg.scene.table_cam.data_types = ["rgb"]
                isaac_env_cfg.scene.wrist_cam.data_types = ["rgb"]
            table_asset = getattr(self.cfg.init_params, "table_asset", None)
            if table_asset:
                from rlinf.envs.isaaclab.scenario_loader import resolve_table_asset_path

                isaac_env_cfg.scene.table.spawn.usd_path = resolve_table_asset_path(
                    str(table_asset),
                    must_exist=True,
                )
            replay_camera_cfg = getattr(self.cfg.init_params, "replay_cameras", None)
            if replay_camera_cfg is not None and getattr(
                replay_camera_cfg, "enabled", False
            ):
                from isaaclab.managers import ObservationTermCfg as ObsTerm
                from isaaclab.managers import SceneEntityCfg
                from isaaclab_tasks.manager_based.manipulation.stack import mdp

                camera_names = list(
                    getattr(
                        replay_camera_cfg, "names", ["replay_cam_0", "replay_cam_1"]
                    )
                )
                height = int(
                    getattr(
                        replay_camera_cfg,
                        "height",
                        self.cfg.init_params.table_cam.height,
                    )
                )
                width = int(
                    getattr(
                        replay_camera_cfg,
                        "width",
                        self.cfg.init_params.table_cam.width,
                    )
                )
                for camera_name in camera_names:
                    camera_cfg = deepcopy(isaac_env_cfg.scene.table_cam)
                    camera_cfg.prim_path = f"{{ENV_REGEX_NS}}/{camera_name}"
                    camera_cfg.height = height
                    camera_cfg.width = width
                    setattr(isaac_env_cfg.scene, camera_name, camera_cfg)
                    setattr(
                        isaac_env_cfg.observations.policy,
                        camera_name,
                        ObsTerm(
                            func=mdp.image,
                            params={
                                "sensor_cfg": SceneEntityCfg(camera_name),
                                "data_type": "rgb",
                                "normalize": False,
                            },
                        ),
                    )
                    if self.enable_depth_observations:
                        setattr(
                            isaac_env_cfg.observations.policy,
                            f"{camera_name}_depth",
                            ObsTerm(
                                func=mdp.image,
                                params={
                                    "sensor_cfg": SceneEntityCfg(camera_name),
                                    "data_type": "distance_to_image_plane",
                                    "normalize": False,
                                },
                            ),
                        )

            scenario_reset_cfg = getattr(self.cfg.init_params, "scenario_reset", None)
            if scenario_reset_cfg is not None and getattr(
                scenario_reset_cfg, "enabled", False
            ):
                from isaaclab.managers import SceneEntityCfg

                from rlinf.envs.isaaclab import custom_events

                isaac_env_cfg.events.randomize_cube_positions.func = (
                    custom_events.apply_scenario_reset
                )
                curriculum_cfg = getattr(scenario_reset_cfg, "curriculum", None)
                curriculum = (
                    OmegaConf.to_container(curriculum_cfg, resolve=True)
                    if curriculum_cfg is not None
                    else None
                )
                isaac_env_cfg.events.randomize_cube_positions.params = {
                    "asset_cfgs": [
                        SceneEntityCfg("cube_1"),
                        SceneEntityCfg("cube_2"),
                        SceneEntityCfg("cube_3"),
                    ],
                    "scenario_file": scenario_reset_cfg.scenario_file,
                    "mode": scenario_reset_cfg.mode,
                    "loop": bool(getattr(scenario_reset_cfg, "loop", True)),
                    "fixed_ids": list(getattr(scenario_reset_cfg, "fixed_ids", [])),
                    "external_group_a_ids": list(
                        getattr(scenario_reset_cfg.external, "group_a_ids", [])
                    ),
                    "external_group_b_ids": list(
                        getattr(scenario_reset_cfg.external, "group_b_ids", [])
                    ),
                    "ratio_a": float(
                        getattr(scenario_reset_cfg.external, "ratio_a", 0.8)
                    ),
                    "worker_rank": int(getattr(self.worker_info, "rank", 0)),
                    "total_workers": int(self.total_num_processes),
                    "envs_per_worker": int(self.num_envs),
                    "seed": int(self.seed),
                    "curriculum": curriculum,
                }
                if hasattr(isaac_env_cfg.events, "randomize_table_visual_material"):
                    isaac_env_cfg.events.randomize_table_visual_material.func = (
                        custom_events.noop_event
                    )
                    isaac_env_cfg.events.randomize_table_visual_material.params = {}

            if getattr(
                self.cfg.init_params,
                "disable_table_visual_randomization",
                False,
            ) and hasattr(isaac_env_cfg.events, "randomize_table_visual_material"):
                from rlinf.envs.isaaclab import custom_events

                isaac_env_cfg.events.randomize_table_visual_material.func = (
                    custom_events.noop_event
                )
                isaac_env_cfg.events.randomize_table_visual_material.params = {}

            if getattr(
                self.cfg.init_params,
                "disable_robot_visual_randomization",
                False,
            ) and hasattr(isaac_env_cfg.events, "randomize_robot_arm_visual_texture"):
                from rlinf.envs.isaaclab import custom_events

                isaac_env_cfg.events.randomize_robot_arm_visual_texture.func = (
                    custom_events.noop_event
                )
                isaac_env_cfg.events.randomize_robot_arm_visual_texture.params = {}

            cube_pose_random_reset_cfg = getattr(
                self.cfg.init_params,
                "cube_pose_random_reset",
                None,
            )
            if (
                (
                    scenario_reset_cfg is None
                    or not getattr(scenario_reset_cfg, "enabled", False)
                )
                and cube_pose_random_reset_cfg is not None
                and getattr(cube_pose_random_reset_cfg, "enabled", False)
            ):
                from isaaclab.managers import SceneEntityCfg
                from isaaclab_tasks.manager_based.manipulation.stack.mdp import (
                    franka_stack_events,
                )

                isaac_env_cfg.events.randomize_cube_positions.func = (
                    franka_stack_events.randomize_object_pose
                )
                isaac_env_cfg.events.randomize_cube_positions.params = {
                    "asset_cfgs": [
                        SceneEntityCfg("cube_1"),
                        SceneEntityCfg("cube_2"),
                        SceneEntityCfg("cube_3"),
                    ],
                    "min_separation": float(
                        getattr(cube_pose_random_reset_cfg, "min_separation", 0.1)
                    ),
                    "pose_range": {
                        "x": tuple(cube_pose_random_reset_cfg.pose_range.x),
                        "y": tuple(cube_pose_random_reset_cfg.pose_range.y),
                        "z": tuple(cube_pose_random_reset_cfg.pose_range.z),
                        "roll": tuple(cube_pose_random_reset_cfg.pose_range.roll),
                        "pitch": tuple(cube_pose_random_reset_cfg.pose_range.pitch),
                        "yaw": tuple(cube_pose_random_reset_cfg.pose_range.yaw),
                    },
                    "max_sample_tries": int(
                        getattr(cube_pose_random_reset_cfg, "max_sample_tries", 5000)
                    ),
                }

            grid_reset_cfg = getattr(self.cfg.init_params, "grid_reset", None)
            if (
                (
                    scenario_reset_cfg is None
                    or not getattr(scenario_reset_cfg, "enabled", False)
                )
                and (
                    cube_pose_random_reset_cfg is None
                    or not getattr(cube_pose_random_reset_cfg, "enabled", False)
                )
                and grid_reset_cfg is not None
                and getattr(grid_reset_cfg, "enabled", False)
            ):
                from isaaclab.managers import SceneEntityCfg

                from rlinf.envs.isaaclab import custom_events

                isaac_env_cfg.events.randomize_cube_positions.func = (
                    custom_events.grid_traverse_object_pose
                )
                isaac_env_cfg.events.randomize_cube_positions.params = {
                    "asset_cfgs": [
                        SceneEntityCfg("cube_1"),
                        SceneEntityCfg("cube_2"),
                        SceneEntityCfg("cube_3"),
                    ],
                    "traverse_asset_idx": grid_reset_cfg.traverse_asset_idx,
                    "x_range": tuple(grid_reset_cfg.x_range),
                    "y_range": tuple(grid_reset_cfg.y_range),
                    "num_x": grid_reset_cfg.num_x,
                    "num_y": grid_reset_cfg.num_y,
                    "z": grid_reset_cfg.z,
                    "yaw": grid_reset_cfg.yaw,
                    "worker_rank": int(getattr(self.worker_info, "rank", 0)),
                    "total_workers": int(self.total_num_processes),
                    "envs_per_worker": int(self.num_envs),
                    "fixed_poses": [
                        [0.4, 0.0, 0.0203, 0.0, 0.0, 0.0],
                        [0.55, 0.05, 0.0203, 0.0, 0.0, 0.0],
                        [0.60, -0.1, 0.0203, 0.0, 0.0, 0.0],
                    ],
                }

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        instruction = [self.task_description] * self.num_envs
        wrist_image = obs["policy"]["wrist_cam"]
        table_image = obs["policy"]["table_cam"]
        quat = obs["policy"]["eef_quat"][
            :, [1, 2, 3, 0]
        ]  # In isaaclab, quat is wxyz not like libero
        states = torch.concatenate(
            [
                obs["policy"]["eef_pos"],
                quat2axisangle_torch(quat),
                obs["policy"]["gripper_pos"],
            ],
            dim=1,
        )

        env_obs = {
            "main_images": table_image,
            "task_descriptions": instruction,
            "states": states,
            "wrist_images": wrist_image,
        }
        if self.enable_depth_observations:
            env_obs["main_images_depth"] = obs["policy"]["table_cam_depth"]
            env_obs["wrist_images_depth"] = obs["policy"]["wrist_cam_depth"]
        return env_obs
