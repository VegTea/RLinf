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

"""Tests for episode-level OpenPI SFT train/test splitting."""

import numpy as np
import pytest
import torch

from rlinf.workers.sft.fsdp_vla_sft_worker import (
    fix_lerobot_episode_data_index,
    split_episode_indices,
)
from rlinf.workers.sft.openpi_action_eval import (
    ActionMetricAccumulator,
    make_fixed_eval_noise,
    select_uniform_chunk_indices,
)


def test_split_episode_indices_holds_out_complete_deterministic_episodes():
    """The wipe-board split must contain 176 train and 20 disjoint test episodes."""
    train_episodes, test_episodes = split_episode_indices(list(range(196)), 20, 0)

    assert len(train_episodes) == 176
    assert len(test_episodes) == 20
    assert set(train_episodes).isdisjoint(test_episodes)
    assert sorted(train_episodes + test_episodes) == list(range(196))
    assert test_episodes == [
        3,
        7,
        13,
        32,
        48,
        55,
        91,
        95,
        105,
        109,
        113,
        115,
        121,
        122,
        140,
        150,
        151,
        171,
        183,
        185,
    ]


@pytest.mark.parametrize("test_episode_count", [0, 3])
def test_split_episode_indices_rejects_empty_train_or_test_split(
    test_episode_count: int,
):
    """Both subsets must retain at least one complete episode."""
    with pytest.raises(ValueError, match="test_episode_count"):
        split_episode_indices([0, 1, 2], test_episode_count, 0)


def test_fix_lerobot_episode_data_index_supports_noncontiguous_episode_ids():
    """Filtered bounds must remain addressable by original episode IDs."""

    class _Dataset:
        episode_data_index = {
            "from": torch.tensor([0, 10, 30]),
            "to": torch.tensor([10, 30, 45]),
        }

    dataset = _Dataset()
    fix_lerobot_episode_data_index(dataset, [3, 7, 188])

    assert dataset.episode_data_index["from"][[3, 7, 188]].tolist() == [0, 10, 30]
    assert dataset.episode_data_index["to"][[3, 7, 188]].tolist() == [10, 30, 45]


def test_uniform_chunk_indices_are_unique_and_avoid_episode_padding():
    """Selected starts are stable, evenly spaced, and have a full horizon."""
    indices, offsets = select_uniform_chunk_indices(
        {3: 24, 7: 20}, [3, 7], action_horizon=15, chunks_per_episode=3
    )

    assert offsets == {3: [0, 4, 9], 7: [0, 2, 5]}
    assert indices == [0, 4, 9, 24, 26, 29]
    assert all(offset <= 24 - 15 for offset in offsets[3])
    assert all(offset <= 20 - 15 for offset in offsets[7])


def test_fixed_eval_noise_is_repeatable_and_sample_specific():
    """Evaluation noise must survive restarts without becoming batch-constant."""
    first = make_fixed_eval_noise(0, 2, 15, 32)
    second = make_fixed_eval_noise(0, 2, 15, 32)

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert not np.array_equal(first[0], first[1])


def test_action_metrics_use_element_weighting_and_integrate_velocity():
    """Decoded metrics cover normalized, native, gripper, and position spaces."""
    prediction = np.zeros((2, 2, 8), dtype=np.float32)
    target = np.ones_like(prediction)
    current_position = np.zeros((2, 7), dtype=np.float32)
    accumulator = ActionMetricAccumulator(control_frequency_hz=2.0)
    accumulator.update(prediction, target, prediction, target, current_position)

    metrics = ActionMetricAccumulator.metrics_from_totals(accumulator.as_totals())
    assert metrics["action_normalized_mse"] == pytest.approx(1.0)
    assert metrics["action_normalized_mae"] == pytest.approx(1.0)
    assert metrics["joint_velocity_mse"] == pytest.approx(1.0)
    assert metrics["gripper_position_mse"] == pytest.approx(1.0)
    # Expert integrated positions are [0.5, 1.0], versus predicted [0, 0].
    assert metrics["joint_position_mse"] == pytest.approx((0.25 + 1.0) / 2)
    assert metrics["joint_position_mae"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["num_chunks"] == 2
