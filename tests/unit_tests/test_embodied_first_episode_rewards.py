# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression checks for first-episode chunk rewards and GAE."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.utils import preprocess_embodied_advantages_inputs
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


@pytest.mark.parametrize("terminal_step", [0, 1, 4, 5, 14, None])
@pytest.mark.parametrize("end_type", ["terminations", "truncations"])
def test_first_episode_rewards_before_chunk_aggregation(terminal_step, end_type):
    actor = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {"rollout_epoch": 2, "reward_type": "chunk_level"},
                "env": {"train": {"auto_reset": False, "ignore_terminations": False}},
            }
        )
    )
    # Two rollout epochs must be independent even if the first ends immediately.
    rewards = torch.arange(1, 31, dtype=torch.float32).reshape(2, 3, 1, 5)
    terminations = torch.zeros(2, 4, 1, 5, dtype=torch.bool)
    truncations = torch.zeros_like(terminations)
    ends = terminations if end_type == "terminations" else truncations
    if terminal_step is not None:
        ends[0, 1 + terminal_step // 5, 0, terminal_step % 5] = True
    ends[1, -1, 0, -1] = True
    batch = {
        "rewards": rewards.reshape(6, 1, 5),
        "dones": (terminations | truncations).reshape(8, 1, 5),
    }
    original_rewards = batch["rewards"].clone()
    processed = EmbodiedFSDPActor._process_received_rollout_batch(actor, batch)

    valid = torch.ones(2, 15, dtype=torch.bool)
    if terminal_step is not None:
        valid[0, terminal_step + 1 :] = False
    step_mask = valid.reshape(2, 3, 5).transpose(0, 1)
    expected_rewards = rewards.squeeze(2).transpose(0, 1).masked_fill(~step_mask, 0)
    torch.testing.assert_close(processed["rewards"], expected_rewards)
    torch.testing.assert_close(batch["rewards"], original_rewards)
    chunk_mask = step_mask.any(dim=-1, keepdim=True)
    assert torch.equal(processed["loss_mask"], chunk_mask)

    inputs = preprocess_embodied_advantages_inputs(
        rewards=processed["rewards"],
        dones=processed["dones"],
        values=torch.zeros(4, 2, 1),
        loss_mask=processed["loss_mask"],
        loss_mask_sum=processed["loss_mask_sum"],
        reward_type="chunk_level",
        adv_type="gae",
    )
    torch.testing.assert_close(inputs["rewards"], expected_rewards.sum(-1))
    _, returns = compute_gae_advantages_and_returns(
        **inputs, gamma=0.99, gae_lambda=0.95, normalize_advantages=False
    )
    if terminal_step is not None:
        terminal_chunk = terminal_step // 5
        # Include terminal reward, but neither reset-episode rewards nor its value.
        assert returns[terminal_chunk, 0] == expected_rewards[terminal_chunk, 0].sum()
        before, _ = compute_gae_advantages_and_returns(**inputs)
        inputs["values"][terminal_chunk + 1 :, 0] = 10000
        after, modified_returns = compute_gae_advantages_and_returns(**inputs)
        mask = inputs["loss_mask"]
        torch.testing.assert_close(before[mask], after[mask])
        torch.testing.assert_close(
            returns[terminal_chunk, 0], modified_returns[terminal_chunk, 0]
        )


@pytest.mark.parametrize(
    ("auto_reset", "ignore_terminations", "reward_type"),
    [
        (True, False, "chunk_level"),
        (False, True, "chunk_level"),
        (False, False, "step_level"),
    ],
)
def test_other_sampling_modes_keep_existing_rewards(
    auto_reset, ignore_terminations, reward_type
):
    actor = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {"rollout_epoch": 1, "reward_type": reward_type},
                "env": {
                    "train": {
                        "auto_reset": auto_reset,
                        "ignore_terminations": ignore_terminations,
                    }
                },
            }
        )
    )
    rewards = torch.ones(1, 1, 5)
    dones = torch.zeros(2, 1, 5, dtype=torch.bool)
    dones[1, 0, 0] = True
    result = EmbodiedFSDPActor._process_received_rollout_batch(
        actor, {"rewards": rewards, "dones": dones}
    )
    torch.testing.assert_close(result["rewards"], rewards)
