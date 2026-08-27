# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RLinf-owned Isaac Lab events for deterministic Stack Cube scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def noop_event(*args, **kwargs) -> None:
    """Disable an inherited Isaac Lab event without patching Isaac Lab."""


def _write_rigid_pose(asset, pose: torch.Tensor, env_id: int, device: str) -> None:
    env_ids = torch.tensor([env_id], device=device, dtype=torch.long)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids
    )


def grid_traverse_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    traverse_asset_idx: int = 1,
    x_range: tuple[float, float] = (0.4, 0.6),
    y_range: tuple[float, float] = (-0.1, 0.1),
    num_x: int = 11,
    num_y: int = 11,
    z: float = 0.0203,
    yaw: float = 0.0,
    fixed_poses: list[list[float]] | None = None,
    worker_rank: int = 0,
    total_workers: int = 1,
    envs_per_worker: int = 1,
) -> None:
    """Traverse one cube over a deterministic grid across all env workers."""
    if env_ids is None:
        return
    fixed_poses = fixed_poses or [
        [0.4, 0.0, 0.0203, 0.0, 0.0, 0.0],
        [0.55, 0.05, 0.0203, 0.0, 0.0, 0.0],
        [0.60, -0.1, 0.0203, 0.0, 0.0, 0.0],
    ]
    if not hasattr(env, "_grid_xy_cache"):
        grid_x = torch.linspace(*x_range, num_x, device=env.device)
        grid_y = torch.linspace(*y_range, num_y, device=env.device)
        env._grid_xy_cache = torch.cartesian_prod(grid_x, grid_y)
        env._grid_xy_batch_index = 0
        env._grid_xy_bootstrap_consumed = False

    local_env_ids = env_ids.to(device=env.device, dtype=torch.long)
    global_batch_size = max(total_workers, 1) * max(envs_per_worker, 1)
    global_ids = (
        env._grid_xy_batch_index * global_batch_size
        + worker_rank * envs_per_worker
        + local_env_ids
    )
    xy_batch = env._grid_xy_cache[global_ids % len(env._grid_xy_cache)]
    if env._grid_xy_bootstrap_consumed:
        env._grid_xy_batch_index += 1
    else:
        env._grid_xy_bootstrap_consumed = True

    for local_idx, env_id in enumerate(local_env_ids.tolist()):
        poses = [pose.copy() for pose in fixed_poses]
        poses[traverse_asset_idx][:2] = xy_batch[local_idx].tolist()
        poses[traverse_asset_idx][2] = z
        poses[traverse_asset_idx][5] = yaw
        for asset_cfg, pose in zip(asset_cfgs, poses):
            pose_tensor = torch.tensor([pose], device=env.device)
            position = pose_tensor[:, :3] + env.scene.env_origins[env_id, :3]
            orientation = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            _write_rigid_pose(
                env.scene[asset_cfg.name],
                torch.cat([position, orientation], dim=-1),
                env_id,
                env.device,
            )


def _scenario_runtime(
    env: ManagerBasedEnv,
    scenario_file: str,
    mode: str,
    loop: bool,
    fixed_ids: list[str] | None,
    external_group_a_ids: list[str] | None,
    external_group_b_ids: list[str] | None,
    ratio_a: float,
    seed: int,
    curriculum: dict | None,
):
    if not hasattr(env, "_scenario_loader"):
        from rlinf.envs.isaaclab.scenario_loader import ScenarioLoader
        from rlinf.envs.isaaclab.scenario_scheduler import ScenarioScheduler

        env._scenario_loader = ScenarioLoader(scenario_file)
        env._scenario_scheduler = ScenarioScheduler(
            scenario_ids=env._scenario_loader.list_ids(),
            mode=mode,
            fixed_ids=fixed_ids,
            external_group_a_ids=external_group_a_ids,
            external_group_b_ids=external_group_b_ids,
            ratio_a=ratio_a,
            loop=loop,
            seed=seed,
            scenario_records=env._scenario_loader.list_records(),
            curriculum=curriculum,
        )
        env._scenario_batch_index = 0
        env._scenario_bootstrap_consumed = False
    return env._scenario_loader, env._scenario_scheduler


def _apply_table_camera_pose(env: ManagerBasedEnv, env_id: int, record: dict) -> None:
    if "table_cam_pos" not in record or "table_cam_rot" not in record:
        return
    camera = env.scene["table_cam"]
    local_pos = torch.tensor(record["table_cam_pos"], device=env.device)
    # Scenario JSONs originate from the Isaac Lab 2 pipeline and store ROS
    # quaternions as (w, x, y, z). Isaac Lab 3 Camera.set_world_poses expects
    # (x, y, z, w), even when ``convention="ros"`` is selected.
    ros_quat_wxyz = torch.tensor(record["table_cam_rot"], device=env.device)
    ros_quat = ros_quat_wxyz[[1, 2, 3, 0]]
    world_pos = local_pos + env.scene.env_origins[env_id, :3]
    camera.set_world_poses(
        positions=world_pos.unsqueeze(0),
        orientations=ros_quat.unsqueeze(0),
        env_ids=[env_id],
        convention="ros",
    )
    camera.reset([env_id])
    if not hasattr(env, "_scenario_last_table_cam_pose_by_env"):
        env._scenario_last_table_cam_pose_by_env = {}
    env._scenario_last_table_cam_pose_by_env[env_id] = {
        "requested_local_pos": record["table_cam_pos"],
        "requested_ros_quat_wxyz": record["table_cam_rot"],
        "applied_ros_quat_xyzw": ros_quat.detach().cpu().tolist(),
        "applied_world_pos": world_pos.detach().cpu().tolist(),
    }


def _reference_asset_paths(prim) -> list[str]:
    paths = []
    try:
        prim_stack = prim.GetPrimStack()
    except Exception:
        return paths
    for prim_spec in prim_stack:
        try:
            references = prim_spec.referenceList.GetAddedOrExplicitItems()
        except Exception:
            continue
        paths.extend(str(ref.assetPath) for ref in references if ref.assetPath)
    return paths


def _same_asset(reference_path: str, target_path: str) -> bool:
    return (
        reference_path == target_path
        or Path(reference_path).name == Path(target_path).name
    )


def _table_prim_path_for_env(env: ManagerBasedEnv, env_id: int) -> str:
    """Resolve a table prim when Isaac Lab records only env_0 for static props."""
    table = env.scene["table"]
    if env_id < len(table.prim_paths):
        return str(table.prim_paths[env_id])

    if not table.prim_paths:
        raise RuntimeError("Scenario reset could not find a table prim")
    if env_id >= len(env.scene.env_prim_paths):
        raise IndexError(f"Scenario reset received invalid environment index: {env_id}")

    source_env_path = env.scene.env_prim_paths[0]
    source_table_path = str(table.prim_paths[0])
    if not source_table_path.startswith(f"{source_env_path}/"):
        raise RuntimeError(
            f"Cannot derive per-environment table prim path from {source_table_path!r}"
        )
    suffix = source_table_path.removeprefix(source_env_path)
    return f"{env.scene.env_prim_paths[env_id]}{suffix}"


def _apply_table_asset(
    env: ManagerBasedEnv, env_id: int, table_asset_name: str
) -> None:
    from isaaclab.sim.utils.stage import get_current_stage

    from rlinf.envs.isaaclab.scenario_loader import resolve_table_asset_path

    target_path = resolve_table_asset_path(table_asset_name, must_exist=True)
    prim_path = _table_prim_path_for_env(env, env_id)
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid table prim for scenario reset: {prim_path}")
    previous = _reference_asset_paths(prim)
    swapped = not any(_same_asset(path, target_path) for path in previous)
    if swapped:
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(target_path)
    if not hasattr(env, "_scenario_last_table_asset_by_env"):
        env._scenario_last_table_asset_by_env = {}
    env._scenario_last_table_asset_by_env[env_id] = {
        "requested": table_asset_name,
        "resolved": target_path,
        "previous_references": previous,
        "active_references": _reference_asset_paths(prim),
        "swapped": swapped,
    }


def apply_scenario_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    scenario_file: str,
    mode: str = "sequential",
    loop: bool = True,
    fixed_ids: list[str] | None = None,
    external_group_a_ids: list[str] | None = None,
    external_group_b_ids: list[str] | None = None,
    ratio_a: float = 0.8,
    worker_rank: int = 0,
    total_workers: int = 1,
    envs_per_worker: int = 1,
    seed: int = 0,
    curriculum: dict | None = None,
) -> None:
    """Apply one JSONL scenario to each selected environment."""
    if env_ids is None:
        return
    local_env_ids = env_ids.to(device=env.device, dtype=torch.long)
    loader, scheduler = _scenario_runtime(
        env,
        scenario_file,
        mode,
        loop,
        fixed_ids,
        external_group_a_ids,
        external_group_b_ids,
        ratio_a,
        seed,
        curriculum,
    )
    global_batch_size = max(total_workers, 1) * max(envs_per_worker, 1)
    global_ids = scheduler.sample_global_batch(
        global_batch_size, env._scenario_batch_index
    )
    offsets = (worker_rank * envs_per_worker + local_env_ids).tolist()
    records = loader.get_by_ids([global_ids[i % len(global_ids)] for i in offsets])
    if env._scenario_bootstrap_consumed:
        env._scenario_batch_index += 1
    else:
        env._scenario_bootstrap_consumed = True

    if not hasattr(env, "_scenario_last_record_by_env"):
        env._scenario_last_record_by_env = {}
    for env_id, record in zip(local_env_ids.tolist(), records):
        env._scenario_last_record_by_env[env_id] = dict(record)
        if record.get("table_asset"):
            _apply_table_asset(env, env_id, record["table_asset"])
        _apply_table_camera_pose(env, env_id, record)
        for asset_index, asset_cfg in enumerate(asset_cfgs, start=1):
            raw_pose = [
                *record[f"cube_{asset_index}_pos"],
                *record[f"cube_{asset_index}_rpy"],
            ]
            pose = torch.tensor([raw_pose], device=env.device)
            position = pose[:, :3] + env.scene.env_origins[env_id, :3]
            orientation = math_utils.quat_from_euler_xyz(
                pose[:, 3], pose[:, 4], pose[:, 5]
            )
            _write_rigid_pose(
                env.scene[asset_cfg.name],
                torch.cat([position, orientation], dim=-1),
                env_id,
                env.device,
            )


def apply_visual_texture_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    texture: str,
    texture_rotation: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Apply PBR textures to custom SeattleLabTable shader prims."""
    del texture_rotation
    from isaaclab.sim.utils.stage import get_current_stage

    prefix = texture.removesuffix("_BaseColor.png")
    values = {
        "inputs:diffuse_texture": texture,
        "inputs:normalmap_texture": f"{prefix}_N.png",
        "inputs:normal_texture": f"{prefix}_N.png",
        "inputs:orm_texture": f"{prefix}_ORM.png",
        "inputs:occlusionroughnessmetallic_texture": f"{prefix}_ORM.png",
    }
    asset = env.scene[asset_cfg.name]
    target_ids = range(len(asset.prim_paths)) if env_ids is None else env_ids.tolist()
    stage = get_current_stage()
    for env_id in target_ids:
        root = asset.prim_paths[int(env_id)]
        shader_paths = [
            f"{root}/Visuals/Looks/table_base/Shader",
            f"{root}/Visuals/Looks/table_parts/Shader",
        ]
        visuals = stage.GetPrimAtPath(f"{root}/Visuals")
        if visuals and visuals.IsValid() and visuals.IsInstance():
            prototype = visuals.GetPrototype()
            base = prototype.GetPath().pathString
            shader_paths.extend(
                [
                    f"{base}/Looks/table_base/Shader",
                    f"{base}/Looks/table_parts/Shader",
                ]
            )
        for shader_path in dict.fromkeys(shader_paths):
            shader = stage.GetPrimAtPath(shader_path)
            if not shader or not shader.IsValid():
                continue
            for attribute_name, value in values.items():
                attribute = shader.GetAttribute(attribute_name)
                if attribute:
                    attribute.Set(value)
