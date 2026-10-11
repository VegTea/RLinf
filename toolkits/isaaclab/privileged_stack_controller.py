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

"""Simulation-only privileged cube stacking with bounded IK and joint servos.

Run using .venv-isaacsim6 after sourcing starts/source.sh. No learned model.
This waypoint planner does NOT implement collision checking: not for real robots.
"""

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


class PandaIK:
    """Extract the exact USD joint chain and solve joint-limited TCP poses."""

    def __init__(self, usd_path: str, margin: float = 0.06):
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(usd_path)
        joints = {
            p.GetName(): UsdPhysics.Joint(p)
            for p in stage.Traverse()
            if p.IsA(UsdPhysics.Joint)
        }
        self.frames = []
        limits = []
        for i in range(1, 8):
            j = joints[f"panda_joint{i}"]
            r = UsdPhysics.RevoluteJoint(j.GetPrim())
            if r.GetAxisAttr().Get() != "X":
                raise ValueError("Expected Panda USD joint-local X axes")
            self.frames.append(self.frames_for_joint(j))
            limits.append([r.GetLowerLimitAttr().Get(), r.GetUpperLimitAttr().Get()])
        self.lo, self.hi = np.deg2rad(limits).T
        self.lo += margin
        self.hi -= margin
        a, b = self.frames_for_joint(joints["panda_hand_joint"])
        self.tool = a @ b
        offset = np.eye(4)
        offset[2, 3] = 0.107
        self.tool = self.tool @ offset

    @staticmethod
    def frames_for_joint(joint) -> tuple[np.ndarray, np.ndarray]:
        """Parent and inverse child joint-frame transforms."""
        frames = []
        for pos, quat in [
            (joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
        ]:
            t = np.eye(4)
            t[:3, :3] = Rotation.from_quat(
                [*quat.GetImaginary(), quat.GetReal()]
            ).as_matrix()
            t[:3, 3] = pos
            frames.append(t)
        return frames[0], np.linalg.inv(frames[1])

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return base-to-TCP transform for seven arm joint angles."""
        t = np.eye(4)
        for angle, (a, b) in zip(q, self.frames):
            r = np.eye(4)
            r[:3, :3] = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
            t = t @ a @ r @ b
        return t @ self.tool

    def solve(
        self, position: np.ndarray, rotation: np.ndarray, seed: np.ndarray
    ) -> np.ndarray:
        """Locally track a Cartesian waypoint within joint limits, or fail."""

        def residual(q):
            t = self.fk(q)
            return np.r_[
                t[:3, 3] - position,
                0.2 * Rotation.from_matrix(rotation.T @ t[:3, :3]).as_rotvec(),
                0.0005 * (q - seed),
            ]

        fit = least_squares(
            residual,
            np.clip(seed, self.lo, self.hi),
            bounds=(self.lo, self.hi),
            max_nfev=180,
        )
        t = self.fk(fit.x)
        error = np.linalg.norm(t[:3, 3] - position)
        angle = np.linalg.norm(Rotation.from_matrix(rotation.T @ t[:3, :3]).as_rotvec())
        if error > 0.003 or angle > 0.03:
            raise RuntimeError(
                f"Bounded IK failed: position={position}, error={error:.4f}m/{angle:.4f}rad"
            )
        return fit.x


def placement_rotation(
    tcp_rotation: np.ndarray, cube_rotation: np.ndarray, yaw: float
) -> np.ndarray:
    """Choose TCP orientation that makes the rigidly held cube upright."""
    cube_in_tcp = tcp_rotation.T @ cube_rotation
    return Rotation.from_euler("z", yaw).as_matrix() @ cube_in_tcp.T


def main() -> None:
    """Run one explicit scenario, save video, trace and success verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-tilt", type=float, default=35.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    assets = Path(os.environ["ISAACSIM_ASSET_ROOT"])
    usd = assets / "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
    if not usd.exists():
        usd = assets / "IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
    if args.self_test:
        ik = PandaIK(str(usd))
        assert np.allclose(ik.fk(np.zeros(7))[:3, 3], [0.088, 0, 0.819], atol=1e-5)
        q = np.array([0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952])
        t = ik.fk(q)
        solved = ik.solve(t[:3, 3], t[:3, :3], q)
        assert np.linalg.norm(ik.fk(solved) - t) < 1e-4
        assert np.all(solved >= ik.lo) and np.all(solved <= ik.hi)
        tcp_rot = Rotation.from_euler("xyz", [0.4, 0.5, 0.6]).as_matrix()
        cube_rot = Rotation.from_euler("xyz", [-0.1, 0.2, 0.3]).as_matrix()
        desired = placement_rotation(tcp_rot, cube_rot, 0.7)
        assert np.allclose(
            desired @ tcp_rot.T @ cube_rot, Rotation.from_euler("z", 0.7).as_matrix()
        )
        print("FK / bounded IK / grasp transform self-check passed", flush=True)
        return

    import imageio.v2 as imageio
    import torch
    from omegaconf import OmegaConf

    from rlinf.envs.isaaclab.scenario_export import validate_scenario_file
    from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv

    if validate_scenario_file(args.scenario) != 1:
        raise ValueError("Expected exactly one scenario")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(
        repo / "examples/embodiment/config/env/isaaclab_stack_cube.yaml"
    )
    cfg.init_params.num_envs = 1
    cfg.init_params.isaaclab_episode_length_steps = 1800
    cfg.init_params.enable_depth_observations = False
    cfg.init_params.antialiasing_mode = "DLAA"
    cfg.init_params.scenario_reset = {
        "enabled": True,
        "mode": "sequential",
        "loop": True,
        "scenario_file": str(args.scenario.resolve()),
        "fixed_ids": [],
        "external": {},
    }
    holder = SimpleNamespace(
        cfg=cfg,
        isaaclab_env_id=cfg.init_params.id,
        seed=42,
        num_envs=1,
        total_num_processes=1,
        worker_info=SimpleNamespace(rank=0),
        enable_depth_observations=False,
    )
    env, app = IsaaclabStackCubeEnv._make_env_function(holder)()
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import cubes_stacked

    ik = PandaIK(str(usd))
    # Preserve the original success predicate below, but do not erase a finished
    # stack by auto-resetting before retreat / stability verification can run.
    success_term = copy.deepcopy(env.termination_manager.get_term_cfg("success"))
    success_term.func = lambda current_env: torch.zeros(
        current_env.num_envs, dtype=torch.bool, device=current_env.device
    )
    success_term.params = {}
    env.termination_manager.set_term_cfg("success", success_term)

    def tensor(value):
        return value.torch if hasattr(value, "torch") else value

    def array(value):
        return tensor(value).detach().cpu().numpy().copy()

    robot = env.scene["robot"]
    arm = env.action_manager.get_term("arm_action")
    env.reset(seed=42)
    origin = array(env.scene.env_origins)[0]
    q_target = array(robot.data.joint_pos)[0, :7]
    actual_tcp, _ = arm._compute_frame_pose()
    if np.linalg.norm(array(actual_tcp)[0] - ik.fk(q_target)[:3, 3]) > 0.005:
        raise RuntimeError("USD FK does not match simulator TCP")

    # Bypass only the unconstrained DLS arm servo; retain physics and gripper term.
    def apply_arm():
        robot.set_joint_position_target_index(
            target=torch.as_tensor(
                q_target[None], device=env.device, dtype=torch.float32
            ),
            joint_ids=arm._joint_ids,
        )

    arm.apply_actions = apply_arm
    video = imageio.get_writer(str(args.output / "video.mp4"), fps=20)
    trace = (args.output / "trace.jsonl").open("x", buffering=1)
    count, phase, grip = 0, "settle", 1.0
    success_once = False

    def cubes():
        return {
            name: array(env.scene[name].data.root_pos_w)[0] - origin
            for name in ["cube_1", "cube_2", "cube_3"]
        }

    def step(n=1):
        nonlocal count, success_once
        for _ in range(n):
            if count >= 1700:
                raise RuntimeError("Controller step budget exceeded")
            action = torch.zeros(1, 7, device=env.device)
            action[0, -1] = grip
            obs, _, terminated, truncated, _ = env.step(action)
            count += 1
            success = bool(cubes_stacked(env).item())
            success_once |= success
            frame = np.concatenate(
                [
                    array(obs["policy"][key])[0, :, :, :3]
                    for key in ["table_cam", "wrist_cam"]
                ],
                axis=1,
            )
            video.append_data(frame.astype(np.uint8))
            trace.write(
                json.dumps(
                    {
                        "step": count,
                        "phase": phase,
                        "success": success,
                        "joint_pos": array(robot.data.joint_pos)[0].tolist(),
                        "joint_target": q_target.tolist(),
                        "tcp_pos": array(arm._compute_frame_pose()[0])[0].tolist(),
                        "cubes": {k: v.tolist() for k, v in cubes().items()},
                        "cube_quats_xyzw": {
                            k: array(env.scene[k].data.root_quat_w)[0].tolist()
                            for k in cubes()
                        },
                        "tcp_quat_xyzw": array(arm._compute_frame_pose()[1])[
                            0
                        ].tolist(),
                    }
                )
                + "\n"
            )
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("Unexpected simulator reset during plan")

    def move(position, rotation, label):
        nonlocal q_target, phase
        phase = label
        actual = array(robot.data.joint_pos)[0, :7]
        initial = ik.fk(actual)
        distance = np.linalg.norm(position - initial[:3, 3])
        angle = np.linalg.norm(
            Rotation.from_matrix(initial[:3, :3].T @ rotation).as_rotvec()
        )
        segments = max(2, int(np.ceil(max(distance / 0.012, angle / 0.06))))
        slerp = Slerp(
            [0, 1], Rotation.from_matrix(np.array([initial[:3, :3], rotation]))
        )
        # Preflight the entire Cartesian segment before moving anything.
        plan = []
        seed = actual
        for f in np.linspace(0, 1, segments + 1)[1:]:
            seed = ik.solve(
                initial[:3, 3] * (1 - f) + position * f, slerp(f).as_matrix(), seed
            )
            plan.append(seed)
        print(f"{label}: {len(plan)} waypoints -> {position.tolist()}", flush=True)
        for target in plan:
            subdivisions = max(
                2, int(np.ceil(np.max(np.abs(target - q_target)) / 0.035))
            )
            begin = q_target.copy()
            for f in np.linspace(0, 1, subdivisions + 1)[1:]:
                q_target = begin * (1 - f) + target * f
                step()
        step(8)
        tcp = array(arm._compute_frame_pose()[0])[0]
        if np.linalg.norm(tcp - position) > 0.015:
            raise RuntimeError(
                f"Tracking error at {label}: actual={tcp}, target={position}"
            )

    result = {
        "controller": "privileged_bounded_ik",
        "model": None,
        "near_tilt_deg": args.near_tilt,
        "seed": 42,
        "step_budget": 1700,
        "scenario": str(args.scenario.resolve()),
        "collision_checked": False,
    }
    (args.output / "initial_state.json").write_text(
        json.dumps({k: v.tolist() for k, v in cubes().items()}, indent=2)
    )
    try:
        step(15)
        down = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        for name, support in [("cube_2", "cube_1"), ("cube_3", "cube_2")]:
            p = cubes()[name]
            near = p[0] < 0.32
            yaw = float(
                Rotation.from_quat(array(env.scene[name].data.root_quat_w)[0]).as_euler(
                    "xyz"
                )[2]
            )
            placement_yaw = yaw
            yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
            rot = (
                Rotation.from_euler(
                    "y", args.near_tilt if near else 0, degrees=True
                ).as_matrix()
                @ Rotation.from_euler("z", yaw).as_matrix()
                @ down
            )
            move(np.array([0.45, 0, 0.30]), down, name + "_safe")
            grasp = p + [0, 0, 0.002]
            pre = grasp - rot[:, 2] * 0.13
            move(pre, rot, name + "_pregrasp")
            move(grasp, rot, name + "_approach")
            phase = name + "_close"
            grip = -1.0
            step(20)
            move(pre, rot, name + "_lift")
            if cubes()[name][2] < p[2] + 0.06:
                raise RuntimeError(f"Grasp/lift failed for {name}")
            # A tilted grasp leaves the cube rotated relative to the TCP.
            # Upright the object, not the gripper; never teleport the cube.
            tcp_rotation = Rotation.from_quat(
                array(arm._compute_frame_pose()[1])[0]
            ).as_matrix()
            cube_rotation = Rotation.from_quat(
                array(env.scene[name].data.root_quat_w)[0]
            ).as_matrix()
            place_rot = placement_rotation(tcp_rotation, cube_rotation, placement_yaw)
            move(np.array([0.45, 0, 0.28]), place_rot, name + "_upright")
            tcp = array(arm._compute_frame_pose()[0])[0]
            offset = cubes()[name] - tcp
            destination = cubes()[support] + [0, 0, 0.0468]
            target = destination - offset
            move(target + [0, 0, 0.12], place_rot, name + "_transfer")
            move(target + [0, 0, 0.005], place_rot, name + "_place")
            phase = name + "_release"
            grip = 1.0
            step(20)
            move(target + [0, 0, 0.14], place_rot, name + "_retreat")
        phase = "verify"
        step(60)
        result.update(
            success=bool(cubes_stacked(env).item()), success_once=success_once
        )
    except Exception as exc:
        result.update(success=False, success_once=success_once, error=str(exc))
        raise
    finally:
        result.update(
            steps=count, final_cubes={k: v.tolist() for k, v in cubes().items()}
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        video.close()
        trace.close()
        env.close()
        app.close()
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
