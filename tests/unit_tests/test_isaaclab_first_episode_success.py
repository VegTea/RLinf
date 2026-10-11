# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for first-episode success across hidden simulator resets."""

from types import SimpleNamespace

import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv
from rlinf.utils.metric_utils import compute_evaluate_metrics


def test_first_episode_success_freezes_inside_chunks_and_resets_selectively():
    env = object.__new__(IsaaclabBaseEnv)
    env.num_envs = 5
    env.device = torch.device("cpu")
    env.worker_info = SimpleNamespace(rank=0)
    env.cfg = SimpleNamespace(max_episode_steps=3)
    env.auto_reset = False
    env.ignore_terminations = False
    env.render_only_at_chunk_end = False
    env.trajectory_recorder = None
    env._episode_ids = torch.zeros(5, dtype=torch.int64)
    env._current_scenario_records = [None] * 5
    env.init_cube_positions = None
    env.prev_step_reward = torch.zeros(5)
    env._elapsed_steps = torch.zeros(5, dtype=torch.int32)
    env._init_metrics()

    # Env 0 fails, 1 succeeds on its terminal step, 2 hits an inner timeout.
    # Env 3 succeeds exactly on the outer timeout; env 4 only succeeds later.
    transitions = iter(
        [
            (
                [0, 1, 0, 0, 0],
                [True, True, False, False, False],
                [False, False, True, False, False],
            ),
            ([1, 0, 1, 0, 0], [False] * 5, [False] * 5),
            ([0, 0, 0, 1, 0], [False] * 5, [False] * 5),
            ([1, 1, 1, 1, 1], [False] * 5, [False] * 5),
            ([1, 0, 1, 0, 1], [True, False, False, False, False], [False] * 5),
        ]
    )

    def simulator_step(*args, **kwargs):
        reward, terminated, truncated = next(transitions)
        return (
            None,
            torch.tensor(reward).float(),
            torch.tensor(terminated),
            torch.tensor(truncated),
            {},
        )

    env.env = SimpleNamespace(step=simulator_step)
    _, _, _, _, infos = env.chunk_step(torch.zeros(5, 4, 7))
    first = infos[0]["episode"]["success_rate"]
    final = infos[-1]["episode"]
    assert first.tolist() == [False, True, False, False, False]
    assert final["success_rate"].tolist() == [False, True, False, True, False]
    assert final["success_once"].tolist() == [True] * 5
    assert env._first_episode_done.all()
    # The existing runner reduction yields an env-slot-weighted success rate.
    shards = [
        {"success_rate": final["success_rate"][:2]},
        {"success_rate": final["success_rate"][2:]},
    ]
    assert abs(float(compute_evaluate_metrics(shards)["success_rate"]) - 0.4) < 1e-6

    env._reset_metrics(torch.tensor([0, 2]))
    assert env._first_episode_done.tolist() == [False, True, False, True, True]
    # Raw termination still closes the window even when output dones are ignored.
    env.ignore_terminations = True
    _, _, terminated, _, info = env.step(return_observation=False)
    assert not terminated.any()
    assert env._first_episode_done.tolist() == [True] * 2 + [False, True, True]
    assert info["episode"]["success_rate"].tolist() == [True, True, True, True, False]
    assert final["success_rate"].tolist() == [False, True, False, True, False]

    env._reset_metrics()
    assert not env._first_episode_done.any()
    assert not env._first_episode_success.any()
    assert not env.success_once.any()
