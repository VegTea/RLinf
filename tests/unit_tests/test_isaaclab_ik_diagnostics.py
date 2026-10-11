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

"""CPU-only check: python tests/unit_tests/test_isaaclab_ik_diagnostics.py."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from rlinf.envs.isaaclab.ik_diagnostics import install_ik_diagnostics


def check_trace(num_envs: int) -> None:
    """Tracing preserves outputs and snapshots tensors before later mutation."""
    q = torch.zeros(num_envs, 7)
    pos = torch.zeros(num_envs, 3)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    jacobian = torch.eye(6, 7).unsqueeze(0).repeat(num_envs, 1, 1)
    calls = []
    controller = NS(
        cfg=NS(ik_method="dls", ik_params={"lambda_val": 0.01}),
        ee_pos_des=pos + 0.01,
        ee_quat_des=quat,
    )

    def compute(*args):
        calls.append(args)
        return q + 0.1

    controller.compute = compute
    arm = NS(
        _ik_controller=controller,
        _joint_names=[str(i) for i in range(7)],
        processed_actions=torch.zeros(num_envs, 6),
        _compute_frame_pose=lambda: (pos, quat),
    )
    data = NS(
        joint_pos=q,
        joint_vel=q,
        root_pos_w=pos,
        joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * 7]).repeat(num_envs, 1, 1),
    )
    robot = NS(data=data, joint_names=arm._joint_names)

    class Scene(dict):
        env_origins = torch.zeros(num_envs, 3)

    scene = Scene(
        robot=robot,
        **{
            name: NS(data=NS(root_pos_w=pos)) for name in ("cube_1", "cube_2", "cube_3")
        },
    )
    result = object()

    def step(action):
        target = controller.compute(pos, quat, jacobian, q)
        assert torch.equal(target, torch.full_like(q, 0.1))
        q.copy_(target)
        return result

    env = NS(
        num_envs=num_envs,
        _scenario_last_record_by_env={i: {"id": f"scene_{i}"} for i in range(num_envs)},
        scene=scene,
        cfg=NS(sim=NS(dt=0.01), decimation=1),
        action_manager=NS(get_term=lambda name: arm, action=torch.zeros(num_envs, 7)),
        step=step,
        close=lambda: None,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        install_ik_diagnostics(env, str(path))
        assert env.step(torch.zeros(num_envs, 7)) is result
        env.close()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["event"] for r in records] == ["metadata"] + ["ik"] * num_envs + [
            "post_step"
        ] * num_envs
        for i in range(num_envs):
            before, after = records[1 + i], records[1 + num_envs + i]
            assert before["env_id"] == after["env_id"] == i
            assert before["scenario_id"] == after["scenario_id"] == f"scene_{i}"
            assert before["arm_joint_pos"] == [0.0] * 7
            assert after["joint_pos_all"][0] > 0
            assert before["jacobian_singular_values"] == [1.0] * 6
        assert len(calls) == 1
        assert torch.equal(
            jacobian, torch.eye(6, 7).unsqueeze(0).repeat(num_envs, 1, 1)
        )


def test_trace_preserves_control() -> None:
    """Check both legacy single-env and batch identity mapping."""
    check_trace(1)
    check_trace(2)


if __name__ == "__main__":
    test_trace_preserves_control()
