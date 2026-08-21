#!/usr/bin/env python3
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

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.envs.isaaclab.replay_utils import (
    VisualScenarioSampler,
    detect_reset_like_tail,
    depth_from_obs,
    depth_from_obs_key,
    discover_npz_files,
    frame_payload,
    image_from_obs_key,
    initial_payload,
    load_replay_trajectory,
    strip_cube_fields,
    table_image_from_obs,
    write_manifest_record,
    write_video,
)
from rlinf.envs.isaaclab.replay_h5 import (
    ReplayH5Config,
    build_goal_labels,
    compute_eef_delta_actions,
    create_h5_datasets,
    load_view_records,
    replay_h5_config_from_omegaconf,
    resolve_cube_color_map,
    select_light_record,
    write_goal_h5,
    write_low_dim_h5,
    write_metadata_h5,
)
from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "config"
    / "isaaclab_franka_stack_cube_trajectory_replay.yaml"
)
DEFAULT_VISUAL_SCENARIO_FILE = (
    REPO_ROOT
    / "rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay IsaacLab stack-cube trajectories saved as .npz files."
    )
    parser.add_argument("trajectory_path", help="A .npz file or directory containing .npz files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Replay yaml path.")
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    parser.add_argument("--mode", choices=("state", "action", "both"), default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Maximum trajectories to replay.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames per trajectory.")
    parser.add_argument("--num-visual-variants", type=int, default=None)
    parser.add_argument(
        "--visual-scenario-file",
        default=None,
        help=(
            "Optional jsonl file for table/camera visual scenario randomization. "
            f"Batch replay defaults to {DEFAULT_VISUAL_SCENARIO_FILE}."
        ),
    )
    parser.add_argument(
        "--visual-samples-per-traj",
        type=int,
        default=None,
        help="Number of random visual scenarios to render per trajectory.",
    )
    parser.add_argument(
        "--visual-scenario-seed",
        type=int,
        default=None,
        help="Seed for deterministic visual scenario sampling.",
    )
    parser.add_argument(
        "--table-asset",
        default=None,
        help="Optional IsaacLab table USD asset name/path to use at env construction time.",
    )
    parser.add_argument("--camera", choices=("table", "wrist"), default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--save-frames", action="store_true", help="Save individual PNG frames.")
    parser.add_argument(
        "--preserve-label-dirs",
        action="store_true",
        help="Write outputs under success/ or fail/ subdirectories when labels are available.",
    )
    parser.add_argument(
        "--keep-suspect-reset-tail",
        action="store_true",
        help="Keep a final frame that appears to be reset-like.",
    )
    parser.add_argument(
        "--reset-jump-threshold",
        type=float,
        default=None,
        help="Cube-position jump threshold for suspect reset-tail detection.",
    )
    parser.add_argument(
        "--action-drift-report",
        action="store_true",
        help="When action replay is enabled, save a state-drift report.",
    )
    parser.add_argument("--save-h5", action="store_true", help="Save replayed state trajectories as H5.")
    parser.add_argument("--h5-output-dir", default=None, help="H5 output directory override.")
    parser.add_argument(
        "--h5-num-views",
        type=int,
        default=None,
        help="Deprecated alias for --h5-num-external-views.",
    )
    parser.add_argument(
        "--h5-num-external-views",
        type=int,
        default=None,
        help="Number of external table-camera views per step.",
    )
    parser.add_argument("--h5-view-scenario-file", default=None, help="Jsonl file containing table_cam_pos/table_cam_rot views.")
    parser.add_argument(
        "--h5-default-view",
        action="store_true",
        help="Use the IsaacLab default table camera instead of a view scenario jsonl.",
    )
    parser.add_argument("--h5-view-selection", choices=("first_n", "random"), default=None)
    parser.add_argument("--h5-view-seed", type=int, default=None)
    parser.add_argument(
        "--h5-use-registered-external-cameras",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use replay-only registered external cameras instead of repeatedly "
            "moving table_cam for each view."
        ),
    )
    parser.add_argument("--h5-light-scenario-file", default=None, help="Jsonl file containing DomeLight settings.")
    parser.add_argument(
        "--h5-default-light",
        action="store_true",
        help="Use the IsaacLab default lighting instead of a light scenario jsonl.",
    )
    parser.add_argument("--h5-light-selection", choices=("random", "sequential"), default=None)
    parser.add_argument("--h5-light-seed", type=int, default=None)
    parser.add_argument("--h5-fixed-light-id", default=None)
    parser.add_argument("--h5-action-type", choices=("eef_delta",), default=None)
    parser.add_argument("--h5-depth-dtype", choices=("float16", "float32"), default=None)
    parser.add_argument("--h5-compression", choices=("lzf", "gzip", "none"), default=None)
    parser.add_argument("--h5-disable-goal", action="store_true", help="Disable success-trajectory goal index generation.")
    parser.add_argument(
        "--h5-cube-color-map-json",
        default=None,
        help='Optional JSON mapping, e.g. {"red":"cube_1","blue":"cube_2","green":"cube_3"}.',
    )
    parser.add_argument(
        "--h5-write-video",
        action="store_true",
        help="Also write state replay videos when --save-h5 is enabled.",
    )
    parser.add_argument("--save-lerobot", action="store_true", help="Save replayed trajectories directly as a LeRobot dataset.")
    parser.add_argument("--lerobot-output-dir", default=None, help="LeRobot dataset output directory.")
    parser.add_argument("--lerobot-repo-id", default="isaaclab-stack-cube-replay")
    parser.add_argument("--lerobot-robot-type", default="franka_panda")
    parser.add_argument("--lerobot-fps", type=int, default=None)
    parser.add_argument(
        "--lerobot-task",
        default="Stack the red block on the blue block, then stack the green block on the red block.",
    )
    parser.add_argument("--lerobot-overwrite", action="store_true")
    parser.add_argument(
        "--lerobot-resume",
        action="store_true",
        help="Append to an existing LeRobot output directory and skip source_replay records already written.",
    )
    parser.add_argument(
        "--lerobot-use-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let LeRobot encode RGB image features as videos.",
    )
    parser.add_argument("--lerobot-image-writer-processes", type=int, default=0)
    parser.add_argument("--lerobot-image-writer-threads", type=int, default=0)
    parser.add_argument("--lerobot-skip-depth", action="store_true")
    parser.add_argument("--lerobot-skip-low-dim-extras", action="store_true")
    parser.add_argument(
        "--lerobot-include-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include goal.step_index and goal.stage_id.",
    )
    parser.add_argument(
        "--lerobot-action-last",
        choices=("zero", "repeat"),
        default="zero",
        help="How to pad the final frame action because eef_delta has T-1 rows.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log per-trajectory replay failures and continue with the next trajectory.",
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Include detailed per-episode replay timing in the manifest/log records.",
    )
    parser.add_argument(
        "--profile-step-interval",
        type=int,
        default=0,
        help="When --profile-timing is set, print per-step timing every N frames.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only initialize/reset the IsaacLab environment, then exit without replaying or writing data.",
    )
    return parser.parse_args()


def load_cfg(args):
    cfg = OmegaConf.load(args.config)
    if args.output_dir is not None:
        cfg.replay.output_dir = args.output_dir
    if args.mode is not None:
        cfg.replay.mode = args.mode
    if args.num_visual_variants is not None:
        cfg.replay.num_visual_variants = args.num_visual_variants
    if args.visual_scenario_file is not None:
        cfg.replay.visual_scenario_file = args.visual_scenario_file
    if args.visual_samples_per_traj is not None:
        cfg.replay.visual_samples_per_traj = args.visual_samples_per_traj
        cfg.replay.num_visual_variants = args.visual_samples_per_traj
    if args.visual_scenario_seed is not None:
        cfg.replay.visual_scenario_seed = args.visual_scenario_seed
    if args.table_asset is not None:
        OmegaConf.update(cfg, "env.init_params.table_asset", args.table_asset, force_add=True)
        OmegaConf.update(
            cfg,
            "env.init_params.disable_table_visual_randomization",
            True,
            force_add=True,
        )
    if (
        cfg.replay.get("visual_scenario_file", None)
        and args.num_visual_variants is None
        and args.visual_samples_per_traj is None
    ):
        cfg.replay.num_visual_variants = int(cfg.replay.get("visual_samples_per_traj", 1))
    if args.camera is not None:
        cfg.replay.camera = args.camera
    if args.fps is not None:
        cfg.replay.fps = args.fps
        cfg.env.video_cfg.fps = args.fps
    if args.reset_jump_threshold is not None:
        cfg.replay.reset_jump_threshold = args.reset_jump_threshold
    if args.keep_suspect_reset_tail:
        cfg.replay.drop_suspect_reset_tail = False
    if args.save_h5:
        OmegaConf.update(cfg, "replay.h5.enabled", True, force_add=True)
    if args.h5_output_dir is not None:
        OmegaConf.update(cfg, "replay.h5.output_dir", args.h5_output_dir, force_add=True)
    h5_num_external_views = args.h5_num_external_views
    if h5_num_external_views is None:
        h5_num_external_views = args.h5_num_views
    if h5_num_external_views is not None:
        OmegaConf.update(cfg, "replay.h5.num_external_views", h5_num_external_views, force_add=True)
    if args.h5_default_view:
        OmegaConf.update(cfg, "replay.h5.view_scenario_file", None, force_add=True)
        OmegaConf.update(cfg, "replay.h5.num_external_views", 1, force_add=True)
        OmegaConf.update(cfg, "replay.h5.camera_names.external", ["table"], force_add=True)
        OmegaConf.update(cfg, "replay.h5.use_registered_external_cameras", False, force_add=True)
        OmegaConf.update(cfg, "env.init_params.replay_cameras.enabled", False, force_add=True)
    elif args.h5_view_scenario_file is not None:
        OmegaConf.update(cfg, "replay.h5.view_scenario_file", args.h5_view_scenario_file, force_add=True)
    if args.h5_view_selection is not None:
        OmegaConf.update(cfg, "replay.h5.view_selection", args.h5_view_selection, force_add=True)
    if args.h5_view_seed is not None:
        OmegaConf.update(cfg, "replay.h5.view_seed", args.h5_view_seed, force_add=True)
    if args.h5_use_registered_external_cameras is not None:
        OmegaConf.update(
            cfg,
            "replay.h5.use_registered_external_cameras",
            bool(args.h5_use_registered_external_cameras),
            force_add=True,
        )
        OmegaConf.update(
            cfg,
            "env.init_params.replay_cameras.enabled",
            bool(args.h5_use_registered_external_cameras),
            force_add=True,
        )
    if args.h5_default_light:
        OmegaConf.update(cfg, "replay.h5.light_scenario_file", None, force_add=True)
    elif args.h5_light_scenario_file is not None:
        OmegaConf.update(cfg, "replay.h5.light_scenario_file", args.h5_light_scenario_file, force_add=True)
    if args.h5_light_selection is not None:
        OmegaConf.update(cfg, "replay.h5.light_selection", args.h5_light_selection, force_add=True)
    if args.h5_light_seed is not None:
        OmegaConf.update(cfg, "replay.h5.light_seed", args.h5_light_seed, force_add=True)
    if args.h5_fixed_light_id is not None:
        OmegaConf.update(cfg, "replay.h5.fixed_light_id", args.h5_fixed_light_id, force_add=True)
    if args.h5_action_type is not None:
        OmegaConf.update(cfg, "replay.h5.action_type", args.h5_action_type, force_add=True)
    if args.h5_depth_dtype is not None:
        OmegaConf.update(cfg, "replay.h5.depth_dtype", args.h5_depth_dtype, force_add=True)
    if args.h5_compression is not None:
        OmegaConf.update(cfg, "replay.h5.compression", args.h5_compression, force_add=True)
    if args.h5_disable_goal:
        OmegaConf.update(cfg, "replay.h5.goal_detection.enabled", False, force_add=True)
    if args.h5_cube_color_map_json is not None:
        OmegaConf.update(
            cfg,
            "replay.h5.cube_color_map",
            json.loads(args.h5_cube_color_map_json),
            force_add=True,
        )
    if args.h5_write_video:
        OmegaConf.update(cfg, "replay.h5.write_video", True, force_add=True)
    if bool(cfg.replay.get("h5", {}).get("use_registered_external_cameras", False)):
        OmegaConf.update(
            cfg,
            "env.init_params.replay_cameras.enabled",
            True,
            force_add=True,
        )
    cfg.replay.preserve_label_dirs = bool(args.preserve_label_dirs)
    OmegaConf.resolve(cfg)
    return cfg


def make_visual_sampler(cfg):
    scenario_file = cfg.replay.get("visual_scenario_file", None)
    if not scenario_file:
        return None
    return VisualScenarioSampler.from_jsonl(
        scenario_file,
        seed=int(cfg.replay.get("visual_scenario_seed", cfg.env.seed)),
    )


def make_env(cfg, seed_offset=0):
    worker_info = SimpleNamespace(rank=0)
    return IsaaclabStackCubeEnv(
        cfg=cfg.env,
        num_envs=int(cfg.env.total_num_envs),
        seed_offset=int(seed_offset),
        total_num_processes=1,
        worker_info=worker_info,
    )


def apply_visual_scenario_if_needed(env, traj, cfg, sampler, variant_idx):
    if sampler is None:
        return None, None
    raw_record = sampler.sample(str(traj.path), variant_idx)
    visual_record = strip_cube_fields(raw_record)
    applied = env.env.replay_apply_visual_scenario(
        {"env_id": 0, "record": visual_record}
    )
    return visual_record, applied


def replay_state_trajectory(env, traj, cfg, output_root, variant_idx, max_frames=None, visual_sampler=None):
    camera = str(cfg.replay.camera)
    fps = int(cfg.replay.fps)
    output_parent = output_root / traj.stem
    if bool(cfg.replay.get("preserve_label_dirs", False)):
        output_parent = output_root / traj.label / traj.stem
    output_dir = output_parent / f"variant_{variant_idx:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env.reset(seed=int(cfg.env.seed) + int(variant_idx))
    visual_record, visual_applied = apply_visual_scenario_if_needed(
        env, traj, cfg, visual_sampler, variant_idx
    )
    tail_info = detect_reset_like_tail(
        traj,
        distance_threshold=float(cfg.replay.reset_jump_threshold),
    )
    frame_count = traj.length
    if bool(cfg.replay.drop_suspect_reset_tail) and tail_info.get("suspect_reset_frame", False):
        frame_count = max(0, frame_count - 1)
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))

    frames = []
    for frame_idx in range(frame_count):
        obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, frame_idx))
        image = table_image_from_obs(obs, camera=camera)
        frames.append(image)
        if cfg.get("save_frames", False):
            image_path = output_dir / "frames" / f"frame_{frame_idx:06d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            import imageio

            imageio.imwrite(image_path, image)

    video_path = output_dir / "state_replay.mp4"
    write_video(frames, video_path, fps=fps)
    return {
        "mode": "state",
        "trajectory": str(traj.path),
        "label": traj.label,
        "variant": int(variant_idx),
        "video": str(video_path),
        "frames": int(len(frames)),
        "tail_check": tail_info,
        "metadata": traj.metadata,
        "visual_scenario": visual_record,
        "visual_scenario_applied": visual_applied,
        "visual_scenario_file": str(visual_sampler.scenario_file) if visual_sampler is not None else None,
    }


def _h5_output_path(output_root: Path, traj, cfg, h5_cfg: ReplayH5Config, variant_idx: int) -> Path:
    h5_root = Path(str(h5_cfg.output_dir)).expanduser().resolve() if h5_cfg.output_dir else output_root / "h5"
    if bool(cfg.replay.get("preserve_label_dirs", False)):
        h5_root = h5_root / traj.label
    return h5_root / f"{traj.stem}_variant{int(variant_idx):03d}.h5"


class ReplayLeRobotWriter:
    def __init__(self, args, cfg):
        if args.lerobot_output_dir is None:
            raise ValueError("--lerobot-output-dir is required when --save-lerobot is set")
        self.output_dir = Path(args.lerobot_output_dir).expanduser().resolve()
        if self.output_dir.exists():
            if args.lerobot_resume:
                pass
            elif not args.lerobot_overwrite:
                raise FileExistsError(
                    f"LeRobot output directory exists. Pass --lerobot-overwrite to replace: {self.output_dir}"
                )
            else:
                shutil.rmtree(self.output_dir)
        self.repo_id = str(args.lerobot_repo_id)
        self.robot_type = str(args.lerobot_robot_type)
        self.fps = int(args.lerobot_fps or cfg.replay.fps)
        self.task = str(args.lerobot_task)
        self.resume = bool(args.lerobot_resume)
        self.use_videos = bool(args.lerobot_use_videos)
        self.image_writer_processes = int(args.lerobot_image_writer_processes)
        self.image_writer_threads = int(args.lerobot_image_writer_threads)
        self.skip_depth = bool(args.lerobot_skip_depth)
        self.skip_low_dim_extras = bool(args.lerobot_skip_low_dim_extras)
        self.include_goal = bool(args.lerobot_include_goal)
        self.action_last = str(args.lerobot_action_last)
        self.dataset = None
        self.episode_count = 0

    def create_if_needed(self, features: dict):
        if self.dataset is not None:
            return
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        if self.resume and (self.output_dir / "meta" / "info.json").is_file():
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.output_dir,
                download_videos=False,
            )
            self.episode_count = int(self.dataset.meta.total_episodes)
            if self.image_writer_processes or self.image_writer_threads:
                self.dataset.start_image_writer(
                    self.image_writer_processes,
                    self.image_writer_threads,
                )
            return

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.output_dir,
            robot_type=self.robot_type,
            fps=self.fps,
            features=features,
            use_videos=self.use_videos,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads,
        )

    def add_frame(self, frame: dict):
        if self.dataset is None:
            raise RuntimeError("LeRobot dataset has not been created")
        self.dataset.add_frame(frame)

    def save_episode(self):
        if self.dataset is None:
            raise RuntimeError("LeRobot dataset has not been created")
        self.dataset.save_episode()
        self.episode_count += 1

    def clear_episode_buffer(self):
        if self.dataset is not None and getattr(self.dataset, "episode_buffer", None) is not None:
            self.dataset.clear_episode_buffer()

    def finalize(self):
        if self.dataset is None:
            return
        image_writer = getattr(self.dataset, "image_writer", None)
        if image_writer is not None:
            image_writer.wait_until_done()
            image_writer.stop()


def _lerobot_feature_schema(
    traj,
    frame_count: int,
    camera_shapes: dict[str, tuple[tuple[int, int, int], tuple[int, int]]],
    h5_cfg: ReplayH5Config,
    writer: ReplayLeRobotWriter,
) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "axis_x",
                    "axis_y",
                    "axis_z",
                    "left_finger",
                    "right_finger",
                ]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "axis_x",
                    "axis_y",
                    "axis_z",
                    "gripper",
                ]
            },
        },
    }
    for camera_name, (rgb_shape, depth_shape) in camera_shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": tuple(rgb_shape),
            "names": ["height", "width", "channels"],
        }
        if not writer.skip_depth:
            features[f"observation.depths.{camera_name}"] = {
                "dtype": "float32",
                "shape": tuple(depth_shape),
                "names": ["height", "width"],
            }
    if not writer.skip_low_dim_extras:
        for key, value in {
            "joint_pos": "observation.joint_pos",
            "joint_vel": "observation.joint_vel",
            "eef_pos": "observation.eef_pos",
            "eef_quat": "observation.eef_quat",
            "gripper_pos": "observation.gripper_pos",
            "cube_positions": "observation.cube_positions",
            "cube_orientations": "observation.cube_orientations",
            "object": "observation.object",
        }.items():
            if key in traj.arrays:
                features[value] = {
                    "dtype": "float32",
                    "shape": tuple(np.asarray(traj.arrays[key][:frame_count]).shape[1:]),
                    "names": [key],
                }
    if writer.include_goal:
        features["goal.step_index"] = {"dtype": "int64", "shape": (1,), "names": None}
        features["goal.stage_id"] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def _quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros((3,), dtype=np.float32)
    q = q / norm
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    sin_half = np.sqrt(max(1.0 - float(q[0] * q[0]), 0.0))
    if sin_half < 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (q[1:4] / sin_half * angle).astype(np.float32)


def _lerobot_state_from_traj(traj, frame_idx: int) -> np.ndarray:
    eef_pos = np.asarray(traj.arrays["eef_pos"][frame_idx], dtype=np.float32)
    eef_quat = np.asarray(traj.arrays["eef_quat"][frame_idx], dtype=np.float32)
    gripper = np.asarray(traj.arrays["gripper_pos"][frame_idx], dtype=np.float32)
    return np.concatenate(
        [eef_pos, _quat_wxyz_to_axis_angle(eef_quat), gripper],
        axis=0,
    ).astype(np.float32)


def _lerobot_action(actions: np.ndarray, frame_idx: int, action_last: str) -> np.ndarray:
    if actions.shape[0] == 0:
        return np.zeros((7,), dtype=np.float32)
    if frame_idx < actions.shape[0]:
        return actions[frame_idx].astype(np.float32)
    if action_last == "repeat":
        return actions[-1].astype(np.float32)
    return np.zeros_like(actions[0], dtype=np.float32)


def _lerobot_low_dim_frame(traj, frame_idx: int) -> dict:
    frame = {}
    for key, output_key in {
        "joint_pos": "observation.joint_pos",
        "joint_vel": "observation.joint_vel",
        "eef_pos": "observation.eef_pos",
        "eef_quat": "observation.eef_quat",
        "gripper_pos": "observation.gripper_pos",
        "cube_positions": "observation.cube_positions",
        "cube_orientations": "observation.cube_orientations",
        "object": "observation.object",
    }.items():
        if key in traj.arrays:
            frame[output_key] = np.asarray(traj.arrays[key][frame_idx], dtype=np.float32)
    return frame


def _lerobot_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if np.isfinite(depth).all():
        return depth
    finite = depth[np.isfinite(depth)]
    max_finite = float(finite.max()) if finite.size else 0.0
    return np.nan_to_num(depth, nan=0.0, posinf=max_finite, neginf=0.0).astype(np.float32, copy=False)


def _lerobot_goal_frame(goal: dict | None, frame_idx: int) -> dict:
    if goal is None:
        return {
            "goal.step_index": np.asarray([-1], dtype=np.int64),
            "goal.stage_id": np.asarray([-1], dtype=np.int64),
        }
    return {
        "goal.step_index": np.asarray([int(goal["goal_step_index"][frame_idx])], dtype=np.int64),
        "goal.stage_id": np.asarray([int(goal["stage_id"][frame_idx])], dtype=np.int64),
    }


def _write_lerobot_source_record(writer: ReplayLeRobotWriter, record: dict):
    meta_dir = writer.output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "source_replay.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True) + "\n")


def _lerobot_record_key(trajectory: str | os.PathLike, variant_idx: int) -> tuple[str, int]:
    return (str(Path(trajectory).expanduser().resolve()), int(variant_idx))


def _load_completed_lerobot_keys(writer: ReplayLeRobotWriter | None) -> set[tuple[str, int]]:
    if writer is None or not writer.resume:
        return set()
    source_replay = writer.output_dir / "meta" / "source_replay.jsonl"
    if not source_replay.is_file():
        return set()
    completed = set()
    with source_replay.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("mode") != "lerobot":
                continue
            trajectory = record.get("trajectory")
            variant = record.get("variant")
            if trajectory is None or variant is None:
                continue
            completed.add(_lerobot_record_key(trajectory, int(variant)))
    return completed


def _timer_add(timing: dict | None, key: str, start: float) -> None:
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + (time.perf_counter() - start)


def _timer_count(timing: dict | None, key: str, value: int = 1) -> None:
    if timing is not None:
        timing[key] = int(timing.get(key, 0)) + int(value)


def _timer_summary(timing: dict | None, frame_count: int) -> dict | None:
    if timing is None:
        return None
    summary = dict(timing)
    frames = max(1, int(frame_count))
    render_keys = (
        "frame_registered_render_s",
        "frame_wrist_render_s",
        "frame_external_camera_set_s",
        "frame_external_render_s",
        "frame_external_extract_s",
        "frame_build_s",
        "frame_add_s",
    )
    summary["frame_loop_accounted_s"] = sum(float(summary.get(key, 0.0)) for key in render_keys)
    for key in render_keys:
        summary[f"{key}_per_frame_ms"] = 1000.0 * float(summary.get(key, 0.0)) / frames
    summary["total_s_per_frame_ms"] = 1000.0 * float(summary.get("total_s", 0.0)) / frames
    return summary


def _registered_external_camera_names(h5_cfg: ReplayH5Config, view_count: int) -> tuple[str, ...]:
    names = tuple(h5_cfg.registered_external_camera_names[: int(view_count)])
    if len(names) < int(view_count):
        raise ValueError(
            "replay.h5.registered_external_camera_names must contain at least "
            f"{view_count} names when use_registered_external_cameras=true."
        )
    return names


def _set_registered_external_cameras(env, h5_cfg: ReplayH5Config, view_records):
    sensor_names = _registered_external_camera_names(h5_cfg, len(view_records))
    return env.env.replay_set_external_cameras(
        {
            "env_id": 0,
            "camera_names": list(sensor_names),
            "records": list(view_records),
        }
    )


def _external_samples_from_obs(obs, h5_cfg: ReplayH5Config, view_count: int):
    return [
        (
            image_from_obs_key(obs, camera_name),
            depth_from_obs_key(obs, f"{camera_name}_depth"),
        )
        for camera_name in _registered_external_camera_names(h5_cfg, view_count)
    ]


def _default_table_samples_from_obs(obs, h5_cfg: ReplayH5Config):
    return [
        (
            table_image_from_obs(obs, camera="table"),
            depth_from_obs(obs, camera="table"),
        )
        for _ in range(int(h5_cfg.num_external_views))
    ]


def replay_h5_trajectory(env, traj, cfg, output_root, variant_idx, max_frames=None, visual_sampler=None):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Saving H5 requires h5py. Use the project .venv Python or install h5py."
        ) from exc

    h5_cfg = replay_h5_config_from_omegaconf(cfg)
    if h5_cfg.view_scenario_file is None and cfg.replay.get("visual_scenario_file", None):
        h5_cfg.view_scenario_file = str(cfg.replay.visual_scenario_file)
    if len(h5_cfg.external_camera_names) < h5_cfg.num_external_views:
        raise ValueError(
            "replay.h5.camera_names.external must contain at least "
            f"{h5_cfg.num_external_views} names."
        )

    h5_path = _h5_output_path(output_root, traj, cfg, h5_cfg, variant_idx)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = h5_path.with_suffix(h5_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    env.reset(seed=int(cfg.env.seed) + int(variant_idx))
    visual_record, visual_applied = apply_visual_scenario_if_needed(
        env, traj, cfg, visual_sampler, variant_idx
    )
    light_record = select_light_record(h5_cfg, str(traj.path), variant_idx)
    light_applied = (
        env.env.replay_apply_light_scenario({"env_id": 0, "record": light_record})
        if light_record is not None
        else None
    )
    tail_info = detect_reset_like_tail(
        traj,
        distance_threshold=float(cfg.replay.reset_jump_threshold),
    )
    frame_count = traj.length
    if bool(cfg.replay.drop_suspect_reset_tail) and tail_info.get("suspect_reset_frame", False):
        frame_count = max(0, frame_count - 1)
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError(f"No frames to write for {traj.path}")

    view_records = load_view_records(h5_cfg, str(traj.path), variant_idx)
    detected_color_map = env.env.replay_get_cube_color_map() if h5_cfg.goal_enabled and traj.label == "success" else None
    color_map = (
        resolve_cube_color_map(h5_cfg.cube_color_map, detected_color_map)
        if h5_cfg.goal_enabled and traj.label == "success"
        else None
    )
    goal = build_goal_labels(traj, color_map, frame_count, h5_cfg) if color_map else None

    external_cameras_applied = (
        _set_registered_external_cameras(env, h5_cfg, view_records)
        if h5_cfg.use_registered_external_cameras
        else None
    )
    first_obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, 0))
    first_wrist_rgb = table_image_from_obs(first_obs, camera="wrist")
    first_wrist_depth = depth_from_obs(first_obs, camera="wrist")
    if h5_cfg.use_registered_external_cameras:
        first_external_samples = _external_samples_from_obs(first_obs, h5_cfg, len(view_records))
    else:
        first_external_samples = []
        for view_record in view_records:
            env.env.replay_set_table_camera({"env_id": 0, "record": view_record})
            obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, 0))
            first_external_samples.append(
                (
                    table_image_from_obs(obs, camera="table"),
                    depth_from_obs(obs, camera="table"),
                )
            )
    camera_names = [
        h5_cfg.wrist_camera_name,
        *h5_cfg.external_camera_names[: len(first_external_samples)],
    ]
    camera_shapes = {
        h5_cfg.wrist_camera_name: (first_wrist_rgb.shape, first_wrist_depth.shape)
    }
    for camera_name, (rgb, depth) in zip(
        h5_cfg.external_camera_names,
        first_external_samples,
        strict=False,
    ):
        camera_shapes[camera_name] = (rgb.shape, depth.shape)

    frames_for_video = [] if h5_cfg.write_video else None
    with h5py.File(tmp_path, "w") as h5_file:
        h5_file.attrs["format"] = "rlinf_isaaclab_replay_h5"
        h5_file.attrs["format_version"] = 1
        datasets = create_h5_datasets(
            h5_file,
            frame_count=frame_count,
            camera_shapes=camera_shapes,
            h5_cfg=h5_cfg,
        )
        write_low_dim_h5(h5_file, traj, frame_count, h5_cfg)
        write_goal_h5(h5_file, goal)
        write_metadata_h5(
            h5_file,
            traj,
            frame_count,
            view_records,
            color_map,
            visual_record,
            visual_applied,
            light_record=light_record,
            light_applied=light_applied,
            h5_cfg=h5_cfg,
        )
        for frame_idx in range(frame_count):
            if frame_idx == 0:
                wrist_rgb = first_wrist_rgb
                wrist_depth = first_wrist_depth
                external_samples = first_external_samples
            else:
                obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, frame_idx))
                wrist_rgb = table_image_from_obs(obs, camera="wrist")
                wrist_depth = depth_from_obs(obs, camera="wrist")
                if h5_cfg.use_registered_external_cameras:
                    external_samples = _external_samples_from_obs(obs, h5_cfg, len(view_records))
                else:
                    external_samples = []
                    for view_record in view_records:
                        env.env.replay_set_table_camera({"env_id": 0, "record": view_record})
                        obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, frame_idx))
                        external_samples.append(
                            (
                                table_image_from_obs(obs, camera="table"),
                                depth_from_obs(obs, camera="table"),
                            )
                        )
            datasets[f"rgb:{h5_cfg.wrist_camera_name}"][frame_idx] = wrist_rgb
            datasets[f"depth:{h5_cfg.wrist_camera_name}"][frame_idx] = wrist_depth.astype(
                h5_cfg.depth_dtype
            )
            for camera_name, (rgb, depth) in zip(
                h5_cfg.external_camera_names,
                external_samples,
                strict=False,
            ):
                datasets[f"rgb:{camera_name}"][frame_idx] = rgb
                datasets[f"depth:{camera_name}"][frame_idx] = depth.astype(h5_cfg.depth_dtype)
            if frames_for_video is not None:
                frames_for_video.append(wrist_rgb)

    os.replace(tmp_path, h5_path)
    video_path = None
    if frames_for_video is not None:
        video_path = h5_path.with_suffix(".mp4")
        write_video(frames_for_video, video_path, fps=int(cfg.replay.fps))

    return {
        "mode": "h5",
        "trajectory": str(traj.path),
        "label": traj.label,
        "variant": int(variant_idx),
        "h5": str(h5_path),
        "video": str(video_path) if video_path is not None else None,
        "frames": int(frame_count),
        "views": int(len(camera_names)),
        "camera_names": camera_names,
        "has_goal": bool(goal is not None),
        "goal_key_steps": goal["key_steps"].astype(int).tolist() if goal else None,
        "tail_check": tail_info,
        "metadata": traj.metadata,
        "visual_scenario": visual_record,
        "visual_scenario_applied": visual_applied,
        "visual_scenario_file": str(visual_sampler.scenario_file) if visual_sampler is not None else None,
        "h5_view_scenario_file": h5_cfg.view_scenario_file,
        "light_scenario": light_record,
        "light_scenario_applied": light_applied,
        "light_scenario_file": h5_cfg.light_scenario_file,
        "registered_external_cameras": external_cameras_applied,
        "cube_color_map": color_map,
        "detected_cube_color_map": detected_color_map,
    }


def replay_lerobot_trajectory(
    env,
    traj,
    cfg,
    variant_idx,
    writer: ReplayLeRobotWriter,
    max_frames=None,
    visual_sampler=None,
    profile_timing=False,
    profile_step_interval=0,
):
    timing = {} if profile_timing else None
    total_start = time.perf_counter()
    h5_cfg = replay_h5_config_from_omegaconf(cfg)
    if h5_cfg.view_scenario_file is None and cfg.replay.get("visual_scenario_file", None):
        h5_cfg.view_scenario_file = str(cfg.replay.visual_scenario_file)
    if len(h5_cfg.external_camera_names) < h5_cfg.num_external_views:
        raise ValueError(
            "replay.h5.camera_names.external must contain at least "
            f"{h5_cfg.num_external_views} names."
        )

    t0 = time.perf_counter()
    env.reset(seed=int(cfg.env.seed) + int(variant_idx))
    _timer_add(timing, "env_reset_s", t0)
    t0 = time.perf_counter()
    visual_record, visual_applied = apply_visual_scenario_if_needed(
        env, traj, cfg, visual_sampler, variant_idx
    )
    _timer_add(timing, "visual_scenario_s", t0)
    t0 = time.perf_counter()
    light_record = select_light_record(h5_cfg, str(traj.path), variant_idx)
    light_applied = (
        env.env.replay_apply_light_scenario({"env_id": 0, "record": light_record})
        if light_record is not None
        else None
    )
    _timer_add(timing, "light_scenario_s", t0)
    t0 = time.perf_counter()
    tail_info = detect_reset_like_tail(
        traj,
        distance_threshold=float(cfg.replay.reset_jump_threshold),
    )
    frame_count = traj.length
    if bool(cfg.replay.drop_suspect_reset_tail) and tail_info.get("suspect_reset_frame", False):
        frame_count = max(0, frame_count - 1)
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError(f"No frames to write for {traj.path}")
    _timer_add(timing, "tail_and_frame_count_s", t0)

    t0 = time.perf_counter()
    view_records = load_view_records(h5_cfg, str(traj.path), variant_idx)
    detected_color_map = env.env.replay_get_cube_color_map() if h5_cfg.goal_enabled and traj.label == "success" else None
    color_map = (
        resolve_cube_color_map(h5_cfg.cube_color_map, detected_color_map)
        if h5_cfg.goal_enabled and traj.label == "success"
        else None
    )
    goal = build_goal_labels(traj, color_map, frame_count, h5_cfg) if color_map else None
    _timer_add(timing, "goal_and_view_setup_s", t0)

    t0 = time.perf_counter()
    use_default_table_camera = not h5_cfg.view_scenario_file
    external_cameras_applied = (
        _set_registered_external_cameras(env, h5_cfg, view_records)
        if h5_cfg.use_registered_external_cameras and not use_default_table_camera
        else None
    )
    _timer_add(timing, "registered_external_camera_set_s", t0)

    t0 = time.perf_counter()
    first_obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, 0))
    first_wrist_rgb = table_image_from_obs(first_obs, camera="wrist")
    first_wrist_depth = depth_from_obs(first_obs, camera="wrist")
    _timer_add(
        timing,
        "first_registered_render_s"
        if h5_cfg.use_registered_external_cameras
        else "first_wrist_render_s",
        t0,
    )
    if use_default_table_camera:
        t0 = time.perf_counter()
        first_external_samples = _default_table_samples_from_obs(first_obs, h5_cfg)
        _timer_add(timing, "first_default_table_extract_s", t0)
    elif h5_cfg.use_registered_external_cameras:
        t0 = time.perf_counter()
        first_external_samples = _external_samples_from_obs(first_obs, h5_cfg, len(view_records))
        _timer_add(timing, "first_external_extract_s", t0)
    else:
        first_external_samples = []
        for view_record in view_records:
            t0 = time.perf_counter()
            env.env.replay_set_table_camera({"env_id": 0, "record": view_record})
            _timer_add(timing, "first_external_camera_set_s", t0)
            t0 = time.perf_counter()
            obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, 0))
            first_external_samples.append(
                (
                    table_image_from_obs(obs, camera="table"),
                    depth_from_obs(obs, camera="table"),
                )
            )
            _timer_add(timing, "first_external_render_s", t0)

    camera_shapes = {
        h5_cfg.wrist_camera_name: (first_wrist_rgb.shape, first_wrist_depth.shape)
    }
    for camera_name, (rgb, depth) in zip(
        h5_cfg.external_camera_names,
        first_external_samples,
        strict=False,
    ):
        camera_shapes[camera_name] = (rgb.shape, depth.shape)
    t0 = time.perf_counter()
    writer.create_if_needed(
        _lerobot_feature_schema(traj, frame_count, camera_shapes, h5_cfg, writer)
    )
    _timer_add(timing, "writer_create_s", t0)

    t0 = time.perf_counter()
    actions = compute_eef_delta_actions(traj, frame_count)
    _timer_add(timing, "action_compute_s", t0)
    camera_names = [
        h5_cfg.wrist_camera_name,
        *h5_cfg.external_camera_names[: len(view_records)],
    ]
    for frame_idx in range(frame_count):
        frame_start = time.perf_counter()
        if frame_idx == 0:
            wrist_rgb = first_wrist_rgb
            wrist_depth = first_wrist_depth
            external_samples = first_external_samples
        else:
            t0 = time.perf_counter()
            obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, frame_idx))
            wrist_rgb = table_image_from_obs(obs, camera="wrist")
            wrist_depth = depth_from_obs(obs, camera="wrist")
            _timer_add(
                timing,
                "frame_registered_render_s"
                if h5_cfg.use_registered_external_cameras
                else "frame_wrist_render_s",
                t0,
            )
            if use_default_table_camera:
                t0 = time.perf_counter()
                external_samples = _default_table_samples_from_obs(obs, h5_cfg)
                _timer_add(timing, "frame_default_table_extract_s", t0)
            elif h5_cfg.use_registered_external_cameras:
                t0 = time.perf_counter()
                external_samples = _external_samples_from_obs(obs, h5_cfg, len(view_records))
                _timer_add(timing, "frame_external_extract_s", t0)
            else:
                external_samples = []
                for view_record in view_records:
                    t0 = time.perf_counter()
                    env.env.replay_set_table_camera({"env_id": 0, "record": view_record})
                    _timer_add(timing, "frame_external_camera_set_s", t0)
                    t0 = time.perf_counter()
                    obs = env.env.replay_set_state_and_get_obs(frame_payload(traj, frame_idx))
                    external_samples.append(
                        (
                            table_image_from_obs(obs, camera="table"),
                            depth_from_obs(obs, camera="table"),
                        )
                    )
                    _timer_add(timing, "frame_external_render_s", t0)

        t0 = time.perf_counter()
        frame = {
            "observation.state": _lerobot_state_from_traj(traj, frame_idx),
            "action": _lerobot_action(actions, frame_idx, writer.action_last),
            "task": writer.task,
            f"observation.images.{h5_cfg.wrist_camera_name}": wrist_rgb.astype(np.uint8),
        }
        if not writer.skip_depth:
            frame[f"observation.depths.{h5_cfg.wrist_camera_name}"] = _lerobot_depth(wrist_depth)
        for camera_name, (rgb, depth) in zip(
            h5_cfg.external_camera_names,
            external_samples,
            strict=False,
        ):
            frame[f"observation.images.{camera_name}"] = rgb.astype(np.uint8)
            if not writer.skip_depth:
                frame[f"observation.depths.{camera_name}"] = _lerobot_depth(depth)
        if not writer.skip_low_dim_extras:
            frame.update(_lerobot_low_dim_frame(traj, frame_idx))
        if writer.include_goal:
            frame.update(_lerobot_goal_frame(goal, frame_idx))
        _timer_add(timing, "frame_build_s", t0)
        t0 = time.perf_counter()
        writer.add_frame(frame)
        _timer_add(timing, "frame_add_s", t0)
        _timer_count(timing, "frames_profiled", 1)
        if profile_timing and profile_step_interval and frame_idx % int(profile_step_interval) == 0:
            print(
                json.dumps(
                    {
                        "mode": "timing_step",
                        "trajectory": str(traj.path),
                        "variant": int(variant_idx),
                        "frame_idx": int(frame_idx),
                        "frame_s": time.perf_counter() - frame_start,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    t0 = time.perf_counter()
    writer.save_episode()
    _timer_add(timing, "save_episode_s", t0)
    if timing is not None:
        timing["total_s"] = time.perf_counter() - total_start
    record = {
        "mode": "lerobot",
        "trajectory": str(traj.path),
        "label": traj.label,
        "variant": int(variant_idx),
        "episode_index": int(writer.episode_count - 1),
        "lerobot_root": str(writer.output_dir),
        "frames": int(frame_count),
        "views": int(len(camera_names)),
        "camera_names": camera_names,
        "has_goal": bool(goal is not None),
        "goal_key_steps": goal["key_steps"].astype(int).tolist() if goal else None,
        "tail_check": tail_info,
        "metadata": traj.metadata,
        "visual_scenario": visual_record,
        "visual_scenario_applied": visual_applied,
        "visual_scenario_file": str(visual_sampler.scenario_file) if visual_sampler is not None else None,
        "h5_view_scenario_file": h5_cfg.view_scenario_file,
        "light_scenario": light_record,
        "light_scenario_applied": light_applied,
        "light_scenario_file": h5_cfg.light_scenario_file,
        "registered_external_cameras": external_cameras_applied,
        "cube_color_map": color_map,
        "detected_cube_color_map": detected_color_map,
    }
    timing_summary = _timer_summary(timing, frame_count)
    if timing_summary is not None:
        record["timing"] = timing_summary
    _write_lerobot_source_record(writer, record)
    return record


def _tensor_action(action, device):
    return torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)


def replay_action_trajectory(
    env,
    traj,
    cfg,
    output_root,
    max_frames=None,
    visual_sampler=None,
    variant_idx=0,
):
    output_parent = output_root / traj.stem
    if bool(cfg.replay.get("preserve_label_dirs", False)):
        output_parent = output_root / traj.label / traj.stem
    if visual_sampler is not None or int(cfg.replay.get("num_visual_variants", 1)) > 1:
        output_parent = output_parent / f"variant_{int(variant_idx):03d}"
    output_dir = output_parent / "action_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    env.reset(seed=int(cfg.env.seed))
    visual_record, visual_applied = apply_visual_scenario_if_needed(
        env, traj, cfg, visual_sampler, variant_idx
    )
    env.env.replay_set_state_and_get_obs(initial_payload(traj))

    observed_objects = []
    frames = []
    actions = traj.arrays.get("action", np.zeros((0, 7), dtype=np.float32))
    if max_frames is not None:
        actions = actions[: int(max_frames)]
    for action in actions:
        obs, _reward, _terminated, _truncated, _info = env.step(
            _tensor_action(action, env.device),
            auto_reset=False,
        )
        raw_obs = env.env.replay_set_state_and_get_obs({})
        policy_obs = raw_obs.get("policy", {})
        if "object" in policy_obs:
            observed_objects.append(np.asarray(policy_obs["object"])[0])
        frames.append(table_image_from_obs(raw_obs, camera=str(cfg.replay.camera)))

    video_path = output_dir / "action_replay.mp4"
    write_video(frames, video_path, fps=int(cfg.replay.fps))

    drift = {}
    if observed_objects and "object" in traj.arrays:
        recorded = traj.arrays["object"][: len(observed_objects)]
        replayed = np.stack(observed_objects, axis=0)
        diff = np.abs(recorded - replayed)
        drift = {
            "object_l1_mean": float(diff.mean()),
            "object_l1_max": float(diff.max()),
            "num_compared_frames": int(diff.shape[0]),
        }
    report = {
        "mode": "action",
        "trajectory": str(traj.path),
        "label": traj.label,
        "variant": int(variant_idx),
        "video": str(video_path),
        "frames": int(len(frames)),
        "drift": drift,
        "metadata": traj.metadata,
        "visual_scenario": visual_record,
        "visual_scenario_applied": visual_applied,
        "visual_scenario_file": str(visual_sampler.scenario_file) if visual_sampler is not None else None,
    }
    with open(output_dir / "drift_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=True, indent=2)
    return report


def main():
    args = parse_args()
    cfg = load_cfg(args)
    cfg.save_frames = bool(args.save_frames)
    mode = str(cfg.replay.mode)
    h5_cfg = replay_h5_config_from_omegaconf(cfg)
    output_root = Path(str(cfg.replay.output_dir)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        env = make_env(cfg)
        try:
            record = {
                "mode": "preflight",
                "status": "ok",
                "registered_external_cameras": bool(
                    cfg.replay.get("h5", {}).get("use_registered_external_cameras", False)
                ),
            }
            print(json.dumps(record, ensure_ascii=True), flush=True)
        finally:
            env.env.close()
        return

    manifest_path = output_root / "manifest.jsonl"
    lerobot_writer = ReplayLeRobotWriter(args, cfg) if args.save_lerobot else None
    completed_lerobot_keys = _load_completed_lerobot_keys(lerobot_writer)

    files = discover_npz_files(args.trajectory_path, max_files=args.max_files)
    visual_sampler = make_visual_sampler(cfg)
    env = make_env(cfg)
    try:
        for file_path in files:
            traj = load_replay_trajectory(file_path)
            if mode in ("state", "both"):
                for variant_idx in range(int(cfg.replay.num_visual_variants)):
                    try:
                        if lerobot_writer is not None:
                            lerobot_key = _lerobot_record_key(file_path, variant_idx)
                            if lerobot_key in completed_lerobot_keys:
                                record = {
                                    "mode": "skip_existing",
                                    "trajectory": str(Path(file_path).expanduser().resolve()),
                                    "variant": int(variant_idx),
                                    "lerobot_root": str(lerobot_writer.output_dir),
                                }
                                write_manifest_record(manifest_path, record)
                                print(json.dumps(record, ensure_ascii=True))
                                if not h5_cfg.enabled and not h5_cfg.write_video:
                                    continue
                            record = replay_lerobot_trajectory(
                                env,
                                traj,
                                cfg,
                                variant_idx,
                                lerobot_writer,
                                max_frames=args.max_frames,
                                visual_sampler=visual_sampler,
                                profile_timing=args.profile_timing,
                                profile_step_interval=args.profile_step_interval,
                            )
                            write_manifest_record(manifest_path, record)
                            print(json.dumps(record, ensure_ascii=True))
                            completed_lerobot_keys.add(lerobot_key)
                            if not h5_cfg.enabled and not h5_cfg.write_video:
                                continue
                        if h5_cfg.enabled:
                            record = replay_h5_trajectory(
                                env,
                                traj,
                                cfg,
                                output_root,
                                variant_idx,
                                max_frames=args.max_frames,
                                visual_sampler=visual_sampler,
                            )
                            write_manifest_record(manifest_path, record)
                            print(json.dumps(record, ensure_ascii=True))
                            if not h5_cfg.write_video:
                                continue
                        record = replay_state_trajectory(
                            env,
                            traj,
                            cfg,
                            output_root,
                            variant_idx,
                            max_frames=args.max_frames,
                            visual_sampler=visual_sampler,
                        )
                        write_manifest_record(manifest_path, record)
                        print(json.dumps(record, ensure_ascii=True))
                    except Exception as exc:
                        if lerobot_writer is not None:
                            lerobot_writer.clear_episode_buffer()
                        if type(exc).__name__ == "InvalidLineError":
                            raise RuntimeError(
                                "LeRobot metadata is not valid JSON. Existing meta/episodes_stats.jsonl "
                                "may contain non-finite values such as Infinity from depth stats, or may "
                                "have been interrupted while writing. Repair or regenerate the affected "
                                "shard before resuming."
                            ) from exc
                        error_record = {
                            "mode": "error",
                            "trajectory": str(file_path),
                            "variant": int(variant_idx),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        write_manifest_record(manifest_path, error_record)
                        print(json.dumps(error_record, ensure_ascii=True), flush=True)
                        if not args.continue_on_error:
                            raise
            if mode in ("action", "both"):
                action_variants = int(cfg.replay.num_visual_variants) if visual_sampler is not None else 1
                for variant_idx in range(action_variants):
                    record = replay_action_trajectory(
                        env,
                        traj,
                        cfg,
                        output_root,
                        max_frames=args.max_frames,
                        visual_sampler=visual_sampler,
                        variant_idx=variant_idx,
                    )
                    write_manifest_record(manifest_path, record)
                    print(json.dumps(record, ensure_ascii=True))
    finally:
        if lerobot_writer is not None:
            lerobot_writer.finalize()
        env.env.close()


if __name__ == "__main__":
    main()
