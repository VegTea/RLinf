# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Franka failure terminations; no changes to reward or reset semantics."""

import math
from typing import Any

import torch


def _tensor(value: Any) -> torch.Tensor:
    """Accept Isaac Lab 3 ProxyArray and legacy Torch data."""
    return value if isinstance(value, torch.Tensor) else value.torch


def _consecutive(env: Any, name: str, hits: torch.Tensor, steps: int) -> torch.Tensor:
    """Count consecutive hits, independently per env (and joint when present)."""
    counts = getattr(env, name, None)
    if counts is None:
        counts = torch.zeros_like(hits, dtype=torch.long)
        setattr(env, name, counts)
    # Both internal and explicit resets clear episode_length_buf. This also
    # clears counters when some *other* termination caused the preceding reset.
    counts[_tensor(env.episode_length_buf) <= 1] = 0
    counts.copy_(torch.where(hits, counts + 1, 0))
    return counts >= steps


def joint_limit_stall(
    env: Any,
    margin: float,
    target_error: float,
    max_velocity: float,
    consecutive_steps: int,
) -> torch.Tensor:
    """Terminate if the same arm joint keeps pushing outward at a hard limit."""
    robot = env.scene["robot"]
    joint_ids = getattr(env, "_safety_arm_joint_ids", None)
    if joint_ids is None:
        joint_ids, _ = robot.find_joints("panda_joint[1-7]")
        if len(joint_ids) != 7:
            raise ValueError("Safety terminations require seven Franka arm joints")
        env._safety_arm_joint_ids = joint_ids
    data = robot.data
    q = _tensor(data.joint_pos)[:, joint_ids]
    v = _tensor(data.joint_vel)[:, joint_ids]
    target = _tensor(data.joint_pos_target)[:, joint_ids]
    limits = _tensor(data.joint_pos_limits)[:, joint_ids]
    outward = ((q <= limits[..., 0] + margin) & (target < q - target_error)) | (
        (q >= limits[..., 1] - margin) & (target > q + target_error)
    )
    stalled = outward & (v.abs() <= max_velocity)
    return _consecutive(env, "_safety_limit_counts", stalled, consecutive_steps).any(
        dim=1
    )


def self_collision(
    env: Any,
    sensor_names: list[str],
    force_threshold: float,
    consecutive_steps: int,
) -> torch.Tensor:
    """Terminate on persistent, filtered robot-to-robot normal contact force."""
    forces = []
    for name in sensor_names:
        matrix = env.scene[name].data.force_matrix_w
        if matrix is None:
            raise RuntimeError(
                f"Self-contact sensor {name} has no filtered force matrix"
            )
        matrix = _tensor(matrix)
        if matrix.numel() == 0:
            raise RuntimeError(f"Self-contact sensor {name} matched no filter bodies")
        forces.append(torch.linalg.vector_norm(matrix, dim=-1).flatten(start_dim=1))
    # Keep counters per pair: alternating contacts must not satisfy persistence.
    hits = torch.cat(forces, dim=1) > force_threshold
    return _consecutive(env, "_safety_contact_counts", hits, consecutive_steps).any(
        dim=1
    )


def franka_contact_filters() -> dict[str, list[str]]:
    """Return non-adjacent body pairs, each sensed once, excluding gripper internals."""
    links = [f"panda_link{i}" for i in range(8)]
    bodies = links + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    ignored = {frozenset(pair) for pair in zip(links, links[1:])}
    ignored.update(
        frozenset(pair)
        for pair in (
            ("panda_link7", "panda_hand"),
            ("panda_link7", "panda_leftfinger"),
            ("panda_link7", "panda_rightfinger"),
            ("panda_hand", "panda_leftfinger"),
            ("panda_hand", "panda_rightfinger"),
            ("panda_leftfinger", "panda_rightfinger"),
        )
    )
    return {
        body: partners
        for i, body in enumerate(bodies)
        if (
            partners := [
                partner
                for partner in bodies[i + 1 :]
                if frozenset((body, partner)) not in ignored
            ]
        )
    }


def configure_safety_terminations(env_cfg: Any, options: Any) -> None:
    """Add the two failure terms and one-body-to-many filtered contact sensors."""
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sensors import ContactSensorCfg

    joint = dict(options.joint_limit)
    contact = dict(options.self_collision)
    for params in (joint, contact):
        for name, value in params.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Safety parameter {name} must be finite and positive")
        steps = params["consecutive_steps"]
        if not isinstance(steps, int):
            raise ValueError("consecutive_steps must be a positive integer")

    robot = env_cfg.scene.robot
    robot.spawn.activate_contact_sensors = True
    robot.spawn.articulation_props.enabled_self_collisions = True
    sensor_names = []
    for body, partners in franka_contact_filters().items():
        name = f"safety_contact_{body}"
        sensor_names.append(name)
        setattr(
            env_cfg.scene,
            name,
            ContactSensorCfg(
                prim_path=f"{robot.prim_path}/{body}",
                filter_prim_paths_expr=[f"{robot.prim_path}/{p}" for p in partners],
                update_period=0.0,
                history_length=0,
            ),
        )
    env_cfg.terminations.joint_limit_stall = TerminationTermCfg(
        func=joint_limit_stall, params=joint
    )
    env_cfg.terminations.self_collision = TerminationTermCfg(
        func=self_collision, params={"sensor_names": sensor_names, **contact}
    )
