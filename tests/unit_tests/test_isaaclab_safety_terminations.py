# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU check: PYTHONPATH=. python tests/unit_tests/test_isaaclab_safety_terminations.py."""

import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rlinf.envs.isaaclab.safety_terminations import (
    configure_safety_terminations,
    franka_contact_filters,
    joint_limit_stall,
    self_collision,
)


def test_safety_terminations() -> None:
    """Check direction, persistence, per-env reset, filtering and configuration."""
    q = torch.zeros(2, 9)
    q[:, 0] = 0.99
    target = q.clone()
    target[:, 0] = 1.1
    data = NS(
        joint_pos=q,
        joint_vel=torch.zeros_like(q),
        joint_pos_target=target,
        joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * 9] * 2),
    )
    robot = NS(data=data, find_joints=lambda pattern: (list(range(7)), []))
    matrix = torch.zeros(2, 1, 2, 3)
    env = NS(
        scene={"robot": robot, "contact": NS(data=NS(force_matrix_w=matrix))},
        episode_length_buf=torch.zeros(2, dtype=torch.long),
    )
    joint_params = {
        "margin": 0.02,
        "target_error": 0.005,
        "max_velocity": 0.05,
        "consecutive_steps": 3,
    }

    def step():
        env.episode_length_buf += 1
        return joint_limit_stall(env, **joint_params).tolist()

    # At the limit but commanded inward: not a stall.
    target[1, 0] = 0.8
    assert step() == [False, False]
    assert step() == [False, False]
    assert step() == [True, False]
    # Partial reset clears only the reset environment's history.
    env.episode_length_buf[0] = 0
    assert step() == [False, False]
    # Moving rapidly and normal gripper closing must not count.
    data.joint_vel[0, 0] = 0.2
    q[:, 7:] = -1
    target[:, 7:] = -2
    for _ in range(4):
        assert step() == [False, False]
    # Lower bound, stationary and outward. ProxyArray path is supported too.
    data.joint_vel.zero_()
    q[:, 0] = -0.99
    target[:, 0] = -1.1
    data.joint_pos = NS(torch=q)
    assert step() == [False, False]
    assert step() == [False, False]
    assert step() == [True, True]
    # Breaking the condition clears persistence.
    target.copy_(q)
    assert step() == [False, False]

    contact_params = {
        "sensor_names": ["contact"],
        "force_threshold": 5.0,
        "consecutive_steps": 2,
    }

    def contact_step():
        env.episode_length_buf += 1
        return self_collision(env, **contact_params).tolist()

    matrix[0, 0, 0, 0] = 6
    assert contact_step() == [False, False]
    assert contact_step() == [True, False]
    env.episode_length_buf[0] = 0
    assert contact_step() == [False, False]
    matrix.zero_()
    assert contact_step() == [False, False]
    # Alternating pairs are not a persistent collision.
    matrix[0, 0, 1, 0] = 6
    assert contact_step() == [False, False]
    matrix.zero_()
    matrix[0, 0, 0, 0] = 6
    assert contact_step() == [False, False]
    matrix[1, 0, 1, 1] = 5  # strict > threshold
    assert contact_step() == [True, False]
    env.scene["contact"].data.force_matrix_w = None
    try:
        contact_step()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing contact data must fail, not disable protection")

    filters = franka_contact_filters()
    pairs = {(a, b) for a, partners in filters.items() for b in partners}
    assert ("panda_link0", "panda_link7") in pairs
    assert ("panda_link3", "panda_leftfinger") in pairs
    assert not any((b, a) in pairs for a, b in pairs)
    assert all(a != b for a, b in pairs)
    assert ("panda_link0", "panda_link1") not in pairs
    assert ("panda_link7", "panda_hand") not in pairs
    assert ("panda_leftfinger", "panda_rightfinger") not in pairs
    assert all("cube" not in a + b and "table" not in a + b for a, b in pairs)

    cfg = NS(
        scene=NS(
            robot=NS(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=NS(
                    activate_contact_sensors=False,
                    articulation_props=NS(enabled_self_collisions=False),
                ),
            )
        ),
        terminations=NS(),
    )
    options = NS(
        joint_limit=joint_params,
        self_collision={
            "force_threshold": 5.0,
            "consecutive_steps": 2,
        },
    )
    # Test wiring without importing/starting Kit on a CPU-only worker.
    with patch.dict(
        sys.modules,
        {
            "isaaclab.managers": NS(TerminationTermCfg=NS),
            "isaaclab.sensors": NS(ContactSensorCfg=NS),
        },
    ):
        configure_safety_terminations(cfg, options)
        assert cfg.scene.robot.spawn.activate_contact_sensors
        assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
        assert cfg.terminations.joint_limit_stall.func is joint_limit_stall
        assert cfg.terminations.self_collision.func is self_collision
        names = cfg.terminations.self_collision.params["sensor_names"]
        assert len(names) == len(filters)
        for name in names:
            sensor = getattr(cfg.scene, name)
            assert sensor.filter_prim_paths_expr
            assert sensor.prim_path not in sensor.filter_prim_paths_expr
        for value in (0, -1, float("nan"), float("inf"), True):
            options.self_collision["force_threshold"] = value
            try:
                configure_safety_terminations(cfg, options)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid threshold {value}")


if __name__ == "__main__":
    test_safety_terminations()
    print("Safety termination checks passed")
