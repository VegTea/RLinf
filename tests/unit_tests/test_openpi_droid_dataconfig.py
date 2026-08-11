# Copyright 2026 The RLinf Authors.
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

"""Tests for the pi0.5-DROID LeRobot input mapping."""

import numpy as np
import pytest
from openpi_client import base_policy

from rlinf.models.embodiment.openpi.dataconfig.droid_dataconfig import (
    DroidJointVelocityInputs,
    JointPositionActionsToVelocity,
    transform_pi05_droid_numeric_episode,
)
from rlinf.models.embodiment.openpi.policies.droid_deployment_policy import (
    DroidAbsoluteJointPositionPolicy,
    joint_velocity_chunk_to_absolute_positions,
)


class _FakeVelocityPolicy(base_policy.BasePolicy):
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.reset_called = False

    def infer(self, obs: dict) -> dict:
        del obs
        return {"actions": self.actions, "policy_timing": {"infer_ms": 1.0}}

    def reset(self) -> None:
        self.reset_called = True


def test_droid_inputs_default_to_right_exterior_and_wrist_images():
    """The default fine-tuning inputs are right exterior, wrist, and padding."""
    left_image = np.full((3, 4, 5), 11, dtype=np.uint8)
    right_image = np.full((3, 4, 5), 22, dtype=np.uint8)
    wrist_image = np.full((3, 4, 5), 33, dtype=np.uint8)
    data = {
        "observation/exterior_image_1_left": left_image,
        "observation/exterior_image_2_left": right_image,
        "observation/wrist_image_left": wrist_image,
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
    }

    inputs = DroidJointVelocityInputs(action_dim=32)(data)

    assert np.array_equal(inputs["image"]["base_0_rgb"], right_image.transpose(1, 2, 0))
    assert not np.array_equal(
        inputs["image"]["base_0_rgb"], left_image.transpose(1, 2, 0)
    )
    assert np.array_equal(
        inputs["image"]["left_wrist_0_rgb"], wrist_image.transpose(1, 2, 0)
    )
    assert np.count_nonzero(inputs["image"]["right_wrist_0_rgb"]) == 0
    assert inputs["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }


def test_numeric_stats_episode_transform_matches_openpi_at_episode_end():
    """The image-free stats path must preserve OpenPI's end padding semantics."""
    joint_positions = np.arange(21, dtype=np.float32).reshape(3, 7) / 10
    gripper_positions = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    absolute_actions = np.concatenate(
        [joint_positions + 0.05, gripper_positions], axis=-1
    )
    action_horizon = 3
    action_dim = 32
    control_frequency_hz = 2.0

    fast_states, fast_actions = transform_pi05_droid_numeric_episode(
        joint_positions,
        gripper_positions,
        absolute_actions,
        action_horizon=action_horizon,
        action_dim=action_dim,
        control_frequency_hz=control_frequency_hz,
    )

    for frame_index in range(len(joint_positions)):
        action_indices = np.minimum(
            frame_index + np.arange(action_horizon), len(joint_positions) - 1
        )
        transformed = DroidJointVelocityInputs(action_dim=action_dim)(
            {
                "observation/state": np.concatenate(
                    [joint_positions[frame_index], gripper_positions[frame_index]]
                ),
                "observation/exterior_image_2_left": np.zeros(
                    (4, 5, 3), dtype=np.uint8
                ),
                "observation/wrist_image_left": np.zeros((4, 5, 3), dtype=np.uint8),
                "actions": absolute_actions[action_indices],
            }
        )
        transformed = JointPositionActionsToVelocity(control_frequency_hz)(transformed)

        np.testing.assert_allclose(fast_states[frame_index], transformed["state"])
        np.testing.assert_allclose(fast_actions[frame_index], transformed["actions"])


def test_velocity_to_position_is_inverse_of_training_action_transform():
    """Deployment integration must invert the SFT velocity-label transform."""
    current_joint_position = np.arange(7, dtype=np.float32) / 10
    target_joint_positions = np.stack(
        [
            current_joint_position + 0.01,
            current_joint_position + 0.03,
            current_joint_position + 0.02,
        ]
    )
    gripper_positions = np.array([[0.1], [0.6], [0.9]], dtype=np.float32)
    absolute_actions = np.concatenate(
        [target_joint_positions, gripper_positions], axis=-1
    )
    frequency = 15.0
    velocity_actions = JointPositionActionsToVelocity(frequency)(
        {
            "state": np.concatenate([current_joint_position, [0.0]]),
            "actions": absolute_actions,
        }
    )["actions"]

    reconstructed = joint_velocity_chunk_to_absolute_positions(
        current_joint_position,
        velocity_actions,
        control_frequency_hz=frequency,
    )

    np.testing.assert_allclose(reconstructed, absolute_actions, atol=1e-6)


def test_action_transform_allows_inference_without_expert_actions():
    """Policy inference observations must not require an action label."""
    data = {"state": np.arange(8, dtype=np.float32)}

    transformed = JointPositionActionsToVelocity(15.0)(data)

    assert transformed is data
    np.testing.assert_array_equal(transformed["state"], data["state"])


def test_deployment_policy_returns_absolute_joint_chunk_and_metadata():
    """The WebSocket-facing wrapper should integrate and preserve timing."""
    velocity_actions = np.zeros((2, 8), dtype=np.float32)
    velocity_actions[:, :7] = 0.15
    velocity_actions[:, 7] = [0.2, 0.8]
    inner_policy = _FakeVelocityPolicy(velocity_actions)
    policy = DroidAbsoluteJointPositionPolicy(
        inner_policy,
        control_frequency_hz=15.0,
    )
    current_joint_position = np.arange(7, dtype=np.float32)

    result = policy.infer({"observation/joint_position": current_joint_position})

    np.testing.assert_allclose(
        result["actions"][:, :7],
        np.stack([current_joint_position + 0.01, current_joint_position + 0.02]),
        atol=1e-6,
    )
    np.testing.assert_allclose(result["actions"][:, 7], [0.2, 0.8])
    assert result["action_space"] == "joint_position"
    assert result["native_model_action_space"] == "joint_velocity"
    assert result["action_horizon"] == 2
    assert result["control_frequency_hz"] == 15.0
    assert result["policy_timing"] == {"infer_ms": 1.0}

    policy.reset()
    assert inner_policy.reset_called


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.nan, np.inf])
def test_deployment_policy_rejects_invalid_frequency(frequency):
    with pytest.raises(ValueError, match="finite and positive"):
        DroidAbsoluteJointPositionPolicy(
            _FakeVelocityPolicy(np.zeros((2, 8), dtype=np.float32)),
            control_frequency_hz=frequency,
        )
