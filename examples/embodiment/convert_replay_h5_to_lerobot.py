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

"""Convert replay H5 trajectories to a LeRobot dataset.

The replay H5 export stores camera datasets with LeRobot-style names such as
``observation.images.wrist``. This script preserves those camera keys and uses
standard LeRobot names for the policy inputs: ``observation.state`` and
``action``.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TASK = "Stack the red block on the blue block, then stack the green block on the red block."

IMAGE_KEYS = (
    "observation.images.wrist",
    "observation.images.cam_high",
    "observation.images.cam_low",
)
DEPTH_KEYS = (
    "observation.depths.wrist",
    "observation.depths.cam_high",
    "observation.depths.cam_low",
)
LOW_DIM_KEYS = {
    "state/joint_pos": "observation.joint_pos",
    "state/joint_vel": "observation.joint_vel",
    "state/eef_pos": "observation.eef_pos",
    "state/eef_quat": "observation.eef_quat",
    "state/gripper_pos": "observation.gripper_pos",
    "state/cube_positions": "observation.cube_positions",
    "state/cube_orientations": "observation.cube_orientations",
    "state/object": "observation.object",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert replay-generated H5 trajectories into a LeRobot dataset."
    )
    parser.add_argument("h5_path", help="A .h5 file or directory containing replay H5 files.")
    parser.add_argument("--output-dir", required=True, help="Output LeRobot dataset root.")
    parser.add_argument(
        "--repo-id",
        default="isaaclab-stack-cube-replay",
        help="LeRobot repo_id stored in dataset metadata.",
    )
    parser.add_argument("--robot-type", default="franka_panda")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let LeRobot encode RGB image features as videos. Depth stays numeric.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Passed to LeRobotDataset.create.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=0,
        help="Passed to LeRobotDataset.create.",
    )
    parser.add_argument(
        "--action-last",
        choices=("zero", "repeat"),
        default="zero",
        help="How to pad the final frame action because H5 eef_delta has T-1 rows.",
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Do not include observation.depths.* features.",
    )
    parser.add_argument(
        "--skip-low-dim-extras",
        action="store_true",
        help="Only write observation.state and action, not individual joint/eef/cube fields.",
    )
    parser.add_argument(
        "--include-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-frame goal.step_index and goal.stage_id, filling -1 when absent.",
    )
    return parser.parse_args()


def discover_h5_files(path: str | Path, max_files: int | None = None) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.rglob("*.h5"))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {root}")
    return files


def h5_get(file_obj: h5py.File, key: str) -> np.ndarray | None:
    if key not in file_obj:
        return None
    return np.asarray(file_obj[key])


def infer_length(file_obj: h5py.File) -> int:
    if "length" in file_obj.attrs:
        return int(file_obj.attrs["length"])
    for key in (*IMAGE_KEYS, "state/eef_pos", "action/eef_delta"):
        if key in file_obj:
            return int(file_obj[key].shape[0])
    raise ValueError("Unable to infer trajectory length from H5 file")


def infer_features(file_path: Path, args) -> dict[str, dict[str, Any]]:
    with h5py.File(file_path, "r") as f:
        length = infer_length(f)
        if args.max_frames is not None:
            length = min(length, int(args.max_frames))
        features: dict[str, dict[str, Any]] = {
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
        for key in IMAGE_KEYS:
            if key in f:
                features[key] = {
                    "dtype": "image",
                    "shape": tuple(f[key].shape[1:]),
                    "names": ["height", "width", "channels"],
                }
        if not args.skip_depth:
            for key in DEPTH_KEYS:
                if key in f:
                    features[key] = {
                        "dtype": "float32",
                        "shape": tuple(f[key].shape[1:]),
                        "names": ["height", "width"],
                    }
        if not args.skip_low_dim_extras:
            for h5_key, out_key in LOW_DIM_KEYS.items():
                if h5_key in f:
                    features[out_key] = {
                        "dtype": "float32",
                        "shape": tuple(f[h5_key].shape[1:]),
                        "names": [out_key.rsplit(".", 1)[-1]],
                    }
        if args.include_goal:
            features["goal.step_index"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }
            features["goal.stage_id"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }
        if length <= 0:
            raise ValueError(f"Empty H5 file: {file_path}")
        return features


def eef_state(file_obj: h5py.File, frame_idx: int) -> np.ndarray:
    eef_pos = np.asarray(file_obj["state/eef_pos"][frame_idx], dtype=np.float32)
    eef_quat = np.asarray(file_obj["state/eef_quat"][frame_idx], dtype=np.float32)
    gripper = np.asarray(file_obj["state/gripper_pos"][frame_idx], dtype=np.float32)
    return np.concatenate(
        [eef_pos, quat_wxyz_to_axis_angle(eef_quat), gripper],
        axis=0,
    ).astype(np.float32)


def frame_action(file_obj: h5py.File, frame_idx: int, length: int, action_last: str) -> np.ndarray:
    action = np.asarray(file_obj["action/eef_delta"], dtype=np.float32)
    if action.shape[0] == 0:
        return np.zeros((7,), dtype=np.float32)
    if frame_idx < action.shape[0]:
        return action[frame_idx].astype(np.float32)
    if action_last == "repeat":
        return action[-1].astype(np.float32)
    return np.zeros_like(action[0], dtype=np.float32)


def quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
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


def frame_from_h5(file_obj: h5py.File, frame_idx: int, length: int, args) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": eef_state(file_obj, frame_idx),
        "action": frame_action(file_obj, frame_idx, length, args.action_last),
        "task": args.task,
    }
    for key in IMAGE_KEYS:
        if key in file_obj:
            frame[key] = np.asarray(file_obj[key][frame_idx], dtype=np.uint8)
    if not args.skip_depth:
        for key in DEPTH_KEYS:
            if key in file_obj:
                frame[key] = np.asarray(file_obj[key][frame_idx], dtype=np.float32)
    if not args.skip_low_dim_extras:
        for h5_key, out_key in LOW_DIM_KEYS.items():
            if h5_key in file_obj:
                frame[out_key] = np.asarray(file_obj[h5_key][frame_idx], dtype=np.float32)
    if args.include_goal:
        if "goal/step_index" in file_obj:
            step_index = int(np.asarray(file_obj["goal/step_index"][frame_idx]))
            stage_id = int(np.asarray(file_obj["goal/stage_id"][frame_idx]))
        else:
            step_index = -1
            stage_id = -1
        frame["goal.step_index"] = np.asarray([step_index], dtype=np.int64)
        frame["goal.stage_id"] = np.asarray([stage_id], dtype=np.int64)
    return frame


def write_source_manifest(output_dir: Path, records: list[dict[str, Any]]) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "source_h5.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    files = discover_h5_files(args.h5_path, args.max_files)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists. Pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = infer_features(files[0], args)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        robot_type=args.robot_type,
        fps=int(args.fps),
        features=features,
        use_videos=bool(args.use_videos),
        image_writer_processes=int(args.image_writer_processes),
        image_writer_threads=int(args.image_writer_threads),
    )

    source_records: list[dict[str, Any]] = []
    try:
        for episode_index, file_path in enumerate(files):
            with h5py.File(file_path, "r") as f:
                length = infer_length(f)
                if args.max_frames is not None:
                    length = min(length, int(args.max_frames))
                for frame_idx in range(length):
                    dataset.add_frame(frame_from_h5(f, frame_idx, length, args))
                dataset.save_episode()
                source_records.append(
                    {
                        "episode_index": episode_index,
                        "source_h5": str(file_path),
                        "length": int(length),
                        "label": str(f.attrs.get("label", "")),
                        "success": bool(f.attrs.get("success", False)),
                        "has_goal": bool(f.attrs.get("has_goal", False)),
                    }
                )
                print(f"saved episode {episode_index:06d}: {file_path} ({length} frames)", flush=True)
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
            dataset.image_writer.stop()

    write_source_manifest(output_dir, source_records)
    print(f"done: {len(source_records)} episodes -> {output_dir}")


if __name__ == "__main__":
    main()
