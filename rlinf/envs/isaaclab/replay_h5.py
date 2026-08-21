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

import numpy as np

from rlinf.envs.isaaclab.replay_utils import (
    IsaacLabReplayTrajectory,
    depth_from_obs,
    frame_payload,
    table_image_from_obs,
)


CUBE_NAMES = ("cube_1", "cube_2", "cube_3")
GOAL_STAGE_NAMES = (
    "pick_red",
    "place_red_on_blue",
    "pick_green",
    "place_green_on_red",
)


@dataclass
class ReplayH5Config:
    enabled: bool = False
    output_dir: str | None = None
    num_external_views: int = 2
    use_registered_external_cameras: bool = False
    registered_external_camera_names: tuple[str, ...] = ("replay_cam_0", "replay_cam_1")
    view_scenario_file: str | None = None
    view_selection: str = "first_n"
    view_seed: int = 0
    wrist_camera_name: str = "wrist"
    external_camera_names: tuple[str, ...] = ("cam_high", "cam_low")
    light_scenario_file: str | None = None
    light_selection: str = "random"
    light_seed: int = 0
    fixed_light_id: str | None = None
    action_type: str = "eef_delta"
    depth_dtype: str = "float16"
    compression: str | None = "lzf"
    compression_opts: int | None = None
    write_video: bool = False
    goal_enabled: bool = True
    cube_color_map: dict[str, str] | None = None
    lift_z_delta: float = 0.035
    stack_xy_threshold: float = 0.035
    stack_z_tolerance: float = 0.025
    eef_cube_distance: float = 0.06
    stable_window: int = 3
    max_velocity: float = 0.02
    cube_size: float = 0.04


def replay_h5_config_from_omegaconf(cfg: Any) -> ReplayH5Config:
    h5_cfg = cfg.replay.get("h5", None)
    if h5_cfg is None:
        h5_cfg = cfg.get("replay_h5", None)
    if h5_cfg is None:
        return ReplayH5Config()

    goal_cfg = h5_cfg.get("goal_detection", {}) if hasattr(h5_cfg, "get") else {}
    compression = h5_cfg.get("compression", "lzf")
    if compression == "none":
        compression = None
    cube_color_map = h5_cfg.get("cube_color_map", None)
    if cube_color_map is not None:
        cube_color_map = dict(cube_color_map)
    return ReplayH5Config(
        enabled=bool(h5_cfg.get("enabled", False)),
        output_dir=h5_cfg.get("output_dir", None),
        num_external_views=int(
            h5_cfg.get("num_external_views", h5_cfg.get("num_views", 2))
        ),
        use_registered_external_cameras=bool(
            h5_cfg.get("use_registered_external_cameras", False)
        ),
        registered_external_camera_names=tuple(
            str(name)
            for name in h5_cfg.get(
                "registered_external_camera_names",
                ["replay_cam_0", "replay_cam_1"],
            )
        ),
        view_scenario_file=h5_cfg.get("view_scenario_file", None),
        view_selection=str(h5_cfg.get("view_selection", "first_n")),
        view_seed=int(h5_cfg.get("view_seed", 0)),
        wrist_camera_name=str(
            h5_cfg.get("camera_names", {}).get("wrist", "wrist")
            if h5_cfg.get("camera_names", None) is not None
            else "wrist"
        ),
        external_camera_names=tuple(
            str(name)
            for name in (
                h5_cfg.get("camera_names", {}).get("external", ["cam_high", "cam_low"])
                if h5_cfg.get("camera_names", None) is not None
                else ["cam_high", "cam_low"]
            )
        ),
        light_scenario_file=h5_cfg.get("light_scenario_file", None),
        light_selection=str(h5_cfg.get("light_selection", "random")),
        light_seed=int(h5_cfg.get("light_seed", h5_cfg.get("view_seed", 0))),
        fixed_light_id=(
            str(h5_cfg.get("fixed_light_id"))
            if h5_cfg.get("fixed_light_id", None) is not None
            else None
        ),
        action_type=str(h5_cfg.get("action_type", "eef_delta")),
        depth_dtype=str(h5_cfg.get("depth_dtype", "float16")),
        compression=compression,
        compression_opts=h5_cfg.get("compression_opts", None),
        write_video=bool(h5_cfg.get("write_video", False)),
        goal_enabled=bool(goal_cfg.get("enabled", True)),
        cube_color_map=cube_color_map,
        lift_z_delta=float(goal_cfg.get("lift_z_delta", 0.035)),
        stack_xy_threshold=float(goal_cfg.get("stack_xy_threshold", 0.035)),
        stack_z_tolerance=float(goal_cfg.get("stack_z_tolerance", 0.025)),
        eef_cube_distance=float(goal_cfg.get("eef_cube_distance", 0.06)),
        stable_window=int(goal_cfg.get("stable_window", 3)),
        max_velocity=float(goal_cfg.get("max_velocity", 0.02)),
        cube_size=float(goal_cfg.get("cube_size", 0.04)),
    )


def load_view_records(h5_cfg: ReplayH5Config, trajectory_key: str, variant_idx: int) -> list[dict[str, Any]]:
    if not h5_cfg.view_scenario_file:
        return [{"id": "current_camera"}]

    path = Path(h5_cfg.view_scenario_file).expanduser().resolve()
    records: list[dict[str, Any]] = []
    seen_camera_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "table_cam_pos" not in record or "table_cam_rot" not in record:
                continue
            camera_id = str(record.get("source_camera_id", record.get("id", line_no)))
            if camera_id in seen_camera_ids:
                continue
            seen_camera_ids.add(camera_id)
            records.append(_camera_only_record(record, fallback_id=camera_id))

    if not records:
        raise ValueError(f"No camera records found in {path}")

    if h5_cfg.view_selection == "random":
        rng = random.Random(f"{h5_cfg.view_seed}:{trajectory_key}:{int(variant_idx)}")
        if len(records) <= h5_cfg.num_external_views:
            selected = list(records)
            rng.shuffle(selected)
        else:
            selected = rng.sample(records, h5_cfg.num_external_views)
    elif h5_cfg.view_selection == "first_n":
        selected = records[: h5_cfg.num_external_views]
    else:
        raise ValueError(f"Unsupported H5 view_selection: {h5_cfg.view_selection}")

    if len(selected) < h5_cfg.num_external_views:
        raise ValueError(
            f"Requested {h5_cfg.num_external_views} views but only found {len(selected)} in {path}"
        )
    return selected


def _camera_only_record(record: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    return {
        "id": str(record.get("source_camera_id", record.get("id", fallback_id))),
        "table_cam_pos": list(record["table_cam_pos"]),
        "table_cam_rot": list(record["table_cam_rot"]),
    }


def load_light_records(light_scenario_file: str | os.PathLike | None) -> list[dict[str, Any]]:
    if not light_scenario_file:
        return []
    path = Path(light_scenario_file).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise KeyError(f"Missing light id at {path}:{line_no}")
            if "light" not in record:
                raise KeyError(f"Missing light payload at {path}:{line_no}")
            records.append(record)
    if not records:
        raise ValueError(f"No light records found in {path}")
    return records


def select_light_record(
    h5_cfg: ReplayH5Config,
    trajectory_key: str,
    variant_idx: int,
) -> dict[str, Any] | None:
    records = load_light_records(h5_cfg.light_scenario_file)
    if not records:
        return None
    if h5_cfg.fixed_light_id is not None:
        for record in records:
            if str(record["id"]) == h5_cfg.fixed_light_id:
                return dict(record)
        raise KeyError(f"Light id not found: {h5_cfg.fixed_light_id}")
    if h5_cfg.light_selection == "random":
        rng = random.Random(f"{h5_cfg.light_seed}:{trajectory_key}:{int(variant_idx)}")
        return dict(records[rng.randrange(len(records))])
    if h5_cfg.light_selection == "sequential":
        return dict(records[int(variant_idx) % len(records)])
    raise ValueError(f"Unsupported H5 light_selection: {h5_cfg.light_selection}")


def resolve_cube_color_map(
    configured_map: dict[str, str] | None,
    detected_map: dict[str, Any] | None,
) -> dict[str, str]:
    if configured_map:
        return _normalize_cube_color_map(configured_map)
    if detected_map and isinstance(detected_map.get("color_to_cube"), dict):
        return _normalize_cube_color_map(detected_map["color_to_cube"])
    raise ValueError(
        "Unable to resolve cube color map for goal detection. Configure "
        "replay.h5.cube_color_map with red/blue/green -> cube_1/cube_2/cube_3."
    )


def _normalize_cube_color_map(color_map: dict[str, str]) -> dict[str, str]:
    required = ("red", "blue", "green")
    normalized = {color: str(color_map[color]) for color in required if color in color_map}
    missing = [color for color in required if color not in normalized]
    if missing:
        raise ValueError(f"cube_color_map missing colors: {missing}")
    invalid = {
        color: cube_name
        for color, cube_name in normalized.items()
        if cube_name not in CUBE_NAMES
    }
    if invalid:
        raise ValueError(f"cube_color_map has invalid cube names: {invalid}")
    return normalized


def build_goal_labels(
    traj: IsaacLabReplayTrajectory,
    color_map: dict[str, str],
    frame_count: int,
    h5_cfg: ReplayH5Config,
) -> dict[str, Any] | None:
    if not h5_cfg.goal_enabled or traj.label != "success":
        return None
    positions = cube_positions_for_detection(traj, frame_count)
    if positions is None:
        raise ValueError(f"Trajectory has no cube position data for goal detection: {traj.path}")
    eef_pos = traj.arrays.get("eef_pos", None)
    if eef_pos is not None:
        eef_pos = np.asarray(eef_pos[:frame_count], dtype=np.float32)

    cube_indices = {name: idx for idx, name in enumerate(CUBE_NAMES)}
    red = positions[:, cube_indices[color_map["red"]]]
    blue = positions[:, cube_indices[color_map["blue"]]]
    green = positions[:, cube_indices[color_map["green"]]]

    pick_red = _detect_pick_step(red, eef_pos, h5_cfg)
    if pick_red["step"] is None:
        return None
    place_red_blue = _detect_stack_step(red, blue, h5_cfg, start=pick_red["step"])
    if place_red_blue["step"] is None:
        return None
    pick_green = _detect_pick_step(green, eef_pos, h5_cfg, start=place_red_blue["step"])
    if pick_green["step"] is None:
        return None
    place_green_red = _detect_stack_step(green, red, h5_cfg, start=pick_green["step"])
    if place_green_red["step"] is None:
        return None

    detections = [pick_red, place_red_blue, pick_green, place_green_red]
    key_steps = np.asarray([int(item["step"]) for item in detections], dtype=np.int32)
    key_steps = np.maximum.accumulate(np.clip(key_steps, 0, frame_count - 1))

    goal_step_index = np.empty(frame_count, dtype=np.int32)
    stage_id = np.empty(frame_count, dtype=np.int8)
    boundaries = [0, int(key_steps[0]), int(key_steps[1]), int(key_steps[2]), frame_count]
    for stage_idx, goal_step in enumerate(key_steps):
        start = boundaries[stage_idx]
        end = boundaries[stage_idx + 1]
        goal_step_index[start:end] = int(goal_step)
        stage_id[start:end] = stage_idx
    if frame_count > 0 and boundaries[-2] < frame_count:
        goal_step_index[boundaries[-2] :] = int(key_steps[-1])
        stage_id[boundaries[-2] :] = len(key_steps) - 1

    return {
        "goal_step_index": goal_step_index,
        "stage_id": stage_id,
        "key_steps": key_steps,
        "stage_names": GOAL_STAGE_NAMES,
        "detection_score": np.asarray([item["score"] for item in detections], dtype=np.float32),
        "detections": detections,
    }


def cube_positions_for_detection(
    traj: IsaacLabReplayTrajectory,
    frame_count: int,
) -> np.ndarray | None:
    if "object" in traj.arrays:
        object_states = np.asarray(traj.arrays["object"][:frame_count], dtype=np.float32)
        if object_states.ndim == 2 and object_states.shape[1] >= 21:
            return object_states[:, :21].reshape(frame_count, 3, 7)[:, :, :3]
    if "cube_positions" in traj.arrays:
        cube_positions = np.asarray(traj.arrays["cube_positions"][:frame_count], dtype=np.float32)
        if cube_positions.ndim == 2 and cube_positions.shape[1] >= 9:
            return cube_positions[:, :9].reshape(frame_count, 3, 3)
    return None


def compute_eef_delta_actions(traj: IsaacLabReplayTrajectory, frame_count: int) -> np.ndarray:
    if frame_count <= 1:
        return np.zeros((0, 7), dtype=np.float32)
    eef_pos = np.asarray(traj.arrays["eef_pos"][:frame_count], dtype=np.float32)
    eef_quat = np.asarray(traj.arrays["eef_quat"][:frame_count], dtype=np.float32)
    gripper = np.asarray(traj.arrays["gripper_pos"][:frame_count], dtype=np.float32)
    pos_delta = eef_pos[1:] - eef_pos[:-1]
    rot_delta = np.stack(
        [_quat_delta_axis_angle(eef_quat[idx], eef_quat[idx + 1]) for idx in range(frame_count - 1)],
        axis=0,
    )
    grip_delta = gripper[1:].mean(axis=1, keepdims=True) - gripper[:-1].mean(axis=1, keepdims=True)
    return np.concatenate([pos_delta, rot_delta, grip_delta], axis=1).astype(np.float32)


def _quat_delta_axis_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    delta = _quat_multiply(q1, _quat_conjugate(q0))
    if delta[0] < 0:
        delta = -delta
    delta = _quat_normalize(delta)
    angle = 2.0 * np.arccos(np.clip(delta[0], -1.0, 1.0))
    sin_half = np.sqrt(max(1.0 - float(delta[0] * delta[0]), 0.0))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = delta[1:4] / sin_half
    return (axis * angle).astype(np.float32)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _detect_pick_step(
    cube_pos: np.ndarray,
    eef_pos: np.ndarray | None,
    cfg: ReplayH5Config,
    start: int = 0,
) -> dict[str, Any]:
    base_z = float(cube_pos[0, 2])
    for idx in range(max(0, int(start)), cube_pos.shape[0]):
        lifted = cube_pos[idx, 2] >= base_z + cfg.lift_z_delta
        close = True
        distance = None
        if eef_pos is not None and idx < eef_pos.shape[0]:
            distance = float(np.linalg.norm(cube_pos[idx] - eef_pos[idx]))
            close = distance <= cfg.eef_cube_distance
        if lifted and close:
            return {"step": int(idx), "score": 1.0, "distance": distance, "best_effort": False}
    return {"step": None, "score": 0.0, "distance": None, "best_effort": False}


def _detect_stack_step(
    top_pos: np.ndarray,
    base_pos: np.ndarray,
    cfg: ReplayH5Config,
    start: int = 0,
) -> dict[str, Any]:
    velocity = np.zeros(top_pos.shape[0], dtype=np.float32)
    if top_pos.shape[0] > 1:
        velocity[1:] = np.linalg.norm(np.diff(top_pos, axis=0), axis=1)
    window = max(1, int(cfg.stable_window))
    for idx in range(max(0, int(start)), top_pos.shape[0] - window + 1):
        top_window = top_pos[idx : idx + window]
        base_window = base_pos[idx : idx + window]
        xy_dist = np.linalg.norm(top_window[:, :2] - base_window[:, :2], axis=1)
        z_gap = top_window[:, 2] - base_window[:, 2]
        stable = velocity[idx : idx + window] <= cfg.max_velocity
        if (
            np.all(xy_dist <= cfg.stack_xy_threshold)
            and np.all(np.abs(z_gap - cfg.cube_size) <= cfg.stack_z_tolerance)
            and np.all(stable)
        ):
            score = float(1.0 / (1.0 + xy_dist.mean() + np.abs(z_gap - cfg.cube_size).mean()))
            return {"step": int(idx), "score": score, "best_effort": False}
    return {"step": None, "score": 0.0, "best_effort": False}


def create_h5_datasets(
    h5_file: Any,
    frame_count: int,
    camera_shapes: dict[str, tuple[tuple[int, int, int], tuple[int, int]]],
    h5_cfg: ReplayH5Config,
) -> dict[str, Any]:
    kwargs = _compression_kwargs(h5_cfg)
    datasets: dict[str, Any] = {}
    for camera_name, (rgb_shape, depth_shape) in camera_shapes.items():
        height, width, channels = rgb_shape
        depth_height, depth_width = depth_shape
        datasets[f"rgb:{camera_name}"] = h5_file.create_dataset(
            f"observation.images.{camera_name}",
            shape=(frame_count, height, width, channels),
            dtype=np.uint8,
            chunks=(1, height, width, channels),
            **kwargs,
        )
        datasets[f"depth:{camera_name}"] = h5_file.create_dataset(
            f"observation.depths.{camera_name}",
            shape=(frame_count, depth_height, depth_width),
            dtype=np.dtype(h5_cfg.depth_dtype),
            chunks=(1, depth_height, depth_width),
            **kwargs,
        )
    return datasets


def write_low_dim_h5(
    h5_file: Any,
    traj: IsaacLabReplayTrajectory,
    frame_count: int,
    h5_cfg: ReplayH5Config,
) -> None:
    state_group = h5_file.create_group("state")
    for key in (
        "joint_pos",
        "joint_vel",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
        "cube_positions",
        "cube_orientations",
        "object",
    ):
        if key in traj.arrays:
            state_group.create_dataset(key, data=np.asarray(traj.arrays[key][:frame_count], dtype=np.float32))

    action_group = h5_file.create_group("action")
    if h5_cfg.action_type == "eef_delta":
        action_group.create_dataset("eef_delta", data=compute_eef_delta_actions(traj, frame_count))
        action_group.attrs["type"] = "eef_delta"
        action_group.attrs["alignment"] = "obs[t] + action[t] -> obs[t+1]"
    else:
        raise ValueError(f"Unsupported H5 action_type: {h5_cfg.action_type}")


def write_goal_h5(h5_file: Any, goal: dict[str, Any] | None) -> None:
    if goal is None:
        h5_file.attrs["has_goal"] = False
        return
    goal_group = h5_file.create_group("goal")
    goal_group.create_dataset("step_index", data=goal["goal_step_index"])
    goal_group.create_dataset("stage_id", data=goal["stage_id"])
    goal_group.create_dataset("key_steps", data=goal["key_steps"])
    goal_group.create_dataset("detection_score", data=goal["detection_score"])
    string_dtype = h5_file.string_dtype(encoding="utf-8") if hasattr(h5_file, "string_dtype") else None
    if string_dtype is None:
        import h5py

        string_dtype = h5py.string_dtype(encoding="utf-8")
    goal_group.create_dataset("stage_names", data=np.asarray(goal["stage_names"], dtype=object), dtype=string_dtype)
    goal_group.attrs["detections_json"] = json.dumps(goal["detections"], ensure_ascii=True)
    h5_file.attrs["has_goal"] = True


def write_metadata_h5(
    h5_file: Any,
    traj: IsaacLabReplayTrajectory,
    frame_count: int,
    view_records: list[dict[str, Any]],
    color_map: dict[str, str] | None,
    visual_record: dict[str, Any] | None,
    visual_applied: dict[str, Any] | None,
    light_record: dict[str, Any] | None = None,
    light_applied: dict[str, Any] | None = None,
    h5_cfg: ReplayH5Config | None = None,
) -> None:
    h5_file.attrs["length"] = int(frame_count)
    h5_file.attrs["source_npz"] = str(traj.path)
    h5_file.attrs["label"] = traj.label
    h5_file.attrs["success"] = bool(traj.label == "success")
    h5_file.attrs["metadata_json"] = json.dumps(traj.metadata, ensure_ascii=True)
    h5_file.attrs["cube_color_map_json"] = json.dumps(color_map or {}, ensure_ascii=True)
    h5_file.attrs["visual_scenario_json"] = json.dumps(visual_record or {}, ensure_ascii=True)
    h5_file.attrs["visual_scenario_applied_json"] = json.dumps(visual_applied or {}, ensure_ascii=True)
    h5_file.attrs["light_scenario_json"] = json.dumps(light_record or {}, ensure_ascii=True)
    h5_file.attrs["light_scenario_applied_json"] = json.dumps(light_applied or {}, ensure_ascii=True)

    meta_group = h5_file.create_group("meta")
    string_dtype = _h5_string_dtype()
    camera_names = []
    if h5_cfg is not None:
        camera_names = [h5_cfg.wrist_camera_name, *h5_cfg.external_camera_names[: len(view_records)]]
    meta_group.create_dataset(
        "camera_names",
        data=np.asarray(camera_names, dtype=object),
        dtype=string_dtype,
    )
    meta_group.create_dataset(
        "view_ids",
        data=np.asarray([str(record.get("id", idx)) for idx, record in enumerate(view_records)], dtype=object),
        dtype=string_dtype,
    )
    if view_records and "table_cam_pos" in view_records[0]:
        meta_group.create_dataset(
            "camera_pos",
            data=np.asarray([record["table_cam_pos"] for record in view_records], dtype=np.float32),
        )
        meta_group.create_dataset(
            "camera_rot",
            data=np.asarray([record["table_cam_rot"] for record in view_records], dtype=np.float32),
        )


def _h5_string_dtype():
    import h5py

    return h5py.string_dtype(encoding="utf-8")


def _compression_kwargs(h5_cfg: ReplayH5Config) -> dict[str, Any]:
    if h5_cfg.compression is None:
        return {}
    kwargs: dict[str, Any] = {"compression": h5_cfg.compression}
    if h5_cfg.compression_opts is not None:
        kwargs["compression_opts"] = h5_cfg.compression_opts
    return kwargs


def write_manifest(path: os.PathLike | str, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True) + "\n")
