# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in per-environment IK tracing without changing controller outputs."""

import json
from pathlib import Path
from typing import Any

import torch


def install_ik_diagnostics(env: Any, output_path: str) -> None:
    """Record actual DLS calls and post-step state to an exclusive JSONL file.

    TCP poses/Jacobians use robot-base coordinates, cubes use environment-local
    coordinates. Quaternions retain IsaacLab's native ordering. SVD uses the
    unscaled geometric Jacobian (mixed linear/angular units), not a universal
    dimensionless singularity score. Each record identifies its local env and
    scenario; each worker must use a distinct output path.
    """
    arm = env.action_manager.get_term("arm_action")
    controller = arm._ik_controller
    robot = env.scene["robot"]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", buffering=1)
    original_compute, original_step, original_close = (
        controller.compute,
        env.step,
        env.close,
    )
    step, substep = 0, 0

    def cpu(value: Any) -> torch.Tensor:
        if hasattr(value, "torch"):
            value = value.torch
        return value.detach().cpu().clone()

    def states() -> list[dict]:
        positions = cpu(robot.data.joint_pos).tolist()
        velocities = cpu(robot.data.joint_vel).tolist()
        origins = cpu(env.scene.env_origins)
        cubes = {
            name: (cpu(env.scene[name].data.root_pos_w) - origins).tolist()
            for name in ("cube_1", "cube_2", "cube_3")
        }
        scenarios = getattr(env, "_scenario_last_record_by_env", {})
        return [
            {
                "env_id": i,
                "scenario_id": scenarios.get(i, {}).get("id"),
                "joint_pos_all": positions[i],
                "joint_vel_all": velocities[i],
                "cube_pos_local": {name: values[i] for name, values in cubes.items()},
            }
            for i in range(env.num_envs)
        ]

    def emit(record: dict) -> None:
        stream.write(json.dumps(record, allow_nan=False) + "\n")

    emit(
        {
            "event": "metadata",
            "num_envs": int(env.num_envs),
            "physics_dt": float(env.cfg.sim.dt),
            "decimation": int(env.cfg.decimation),
            "joint_names": robot.joint_names,
            "arm_joint_names": arm._joint_names,
            "joint_pos_limits": cpu(robot.data.joint_pos_limits)[0].tolist(),
            "joint_pos_limits_per_env": cpu(robot.data.joint_pos_limits).tolist(),
            "base_pos_local": (
                cpu(robot.data.root_pos_w)[0] - cpu(env.scene.env_origins)[0]
            ).tolist(),
            "ik_method": controller.cfg.ik_method,
            "ik_params": controller.cfg.ik_params,
        }
    )

    def compute(ee_pos, ee_quat, jacobian, joint_pos):
        nonlocal substep
        # Observe the existing call; never recompute the action-term Jacobian,
        # whose implementation can mutate a shared simulator tensor in place.
        target = original_compute(ee_pos, ee_quat, jacobian, joint_pos)
        j = cpu(jacobian)
        batch = {
            "action": cpu(env.action_manager.action).tolist(),
            "processed_arm_action": cpu(arm.processed_actions).tolist(),
            "tcp_pos": cpu(ee_pos).tolist(),
            "tcp_quat": cpu(ee_quat).tolist(),
            "target_tcp_pos": cpu(controller.ee_pos_des).tolist(),
            "target_tcp_quat": cpu(controller.ee_quat_des).tolist(),
            "arm_joint_pos": cpu(joint_pos).tolist(),
            "ik_joint_target": cpu(target).tolist(),
            "jacobian": j.tolist(),
            "jacobian_singular_values": torch.linalg.svdvals(j).tolist(),
        }
        for i, state in enumerate(states()):
            emit(
                {
                    "event": "ik",
                    "step": step,
                    "substep": substep,
                    **{key: values[i] for key, values in batch.items()},
                    **state,
                }
            )
        substep += 1
        return target

    def traced_step(action):
        nonlocal step, substep
        step += 1
        substep = 0
        result = original_step(action)
        pos, quat = arm._compute_frame_pose()
        positions, quaternions = cpu(pos).tolist(), cpu(quat).tolist()
        for i, state in enumerate(states()):
            emit(
                {
                    "event": "post_step",
                    "step": step,
                    "tcp_pos": positions[i],
                    "tcp_quat": quaternions[i],
                    **state,
                }
            )
        return result

    def close():
        try:
            return original_close()
        finally:
            stream.close()

    controller.compute = compute
    env.step = traced_step
    env.close = close
