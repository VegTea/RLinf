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

"""Deployment wrappers for the DROID joint-velocity OpenPI policy."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from openpi_client import base_policy
from typing_extensions import override


def joint_velocity_chunk_to_absolute_positions(
    current_joint_position: np.ndarray,
    actions: np.ndarray,
    *,
    control_frequency_hz: float,
) -> np.ndarray:
    """Integrate a DROID velocity chunk into absolute joint targets.

    The first seven action dimensions are joint velocities. Dimension eight is
    an absolute gripper command and is copied without integration.

    Args:
        current_joint_position: Current seven-dimensional robot joint position.
        actions: Action chunk shaped ``(horizon, >=8)``.
        control_frequency_hz: Frequency used to define the velocity labels.

    Returns:
        An ``(horizon, 8)`` chunk containing seven absolute joint targets and
        the unchanged absolute gripper command.

    Raises:
        ValueError: If an input has the wrong shape, contains non-finite values,
            or the control frequency is not positive.
    """
    joint_position = np.asarray(current_joint_position, dtype=np.float64)
    action_chunk = np.asarray(actions)
    if joint_position.shape != (7,):
        raise ValueError(
            "Expected current_joint_position with shape (7,), got "
            f"{joint_position.shape}."
        )
    if action_chunk.ndim != 2 or action_chunk.shape[-1] < 8:
        raise ValueError(
            f"Expected actions with shape (horizon, >=8), got {action_chunk.shape}."
        )
    if not np.isfinite(control_frequency_hz) or control_frequency_hz <= 0:
        raise ValueError(
            "control_frequency_hz must be finite and positive, got "
            f"{control_frequency_hz}."
        )
    if not np.all(np.isfinite(joint_position)) or not np.all(
        np.isfinite(action_chunk[:, :8])
    ):
        raise ValueError("Joint positions and actions must contain finite values.")

    output_dtype = np.result_type(action_chunk.dtype, np.float32)
    absolute_actions = np.asarray(action_chunk[:, :8], dtype=output_dtype).copy()
    joint_velocity = np.asarray(action_chunk[:, :7], dtype=np.float64)
    absolute_actions[:, :7] = joint_position + np.cumsum(
        joint_velocity / control_frequency_hz, axis=0
    )
    if not np.all(np.isfinite(absolute_actions)):
        raise ValueError("Integrated absolute actions must contain finite values.")
    return absolute_actions


class DroidAbsoluteJointPositionPolicy(base_policy.BasePolicy):
    """Expose a DROID velocity policy as an absolute-joint-position policy.

    This wrapper intentionally performs the conversion after the wrapped
    OpenPI policy has unnormalized its output. Each new action chunk is
    integrated from the joint position contained in the corresponding request.
    """

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        *,
        control_frequency_hz: float,
        joint_position_key: str = "observation/joint_position",
    ) -> None:
        if not np.isfinite(control_frequency_hz) or control_frequency_hz <= 0:
            raise ValueError(
                "control_frequency_hz must be finite and positive, got "
                f"{control_frequency_hz}."
            )
        self._policy = policy
        self._control_frequency_hz = control_frequency_hz
        self._joint_position_key = joint_position_key

    @override
    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        if self._joint_position_key not in obs:
            raise KeyError(
                f"Observation is missing {self._joint_position_key!r}, which is "
                "required to integrate joint velocities."
            )
        result = self._policy.infer(dict(obs))
        if "actions" not in result:
            raise KeyError("Wrapped policy response is missing 'actions'.")
        absolute_actions = joint_velocity_chunk_to_absolute_positions(
            np.asarray(obs[self._joint_position_key]),
            np.asarray(result["actions"]),
            control_frequency_hz=self._control_frequency_hz,
        )
        return {
            **result,
            "actions": absolute_actions,
            "action_space": "joint_position",
            "native_model_action_space": "joint_velocity",
            "control_frequency_hz": self._control_frequency_hz,
            "action_horizon": absolute_actions.shape[0],
        }

    @override
    def reset(self) -> None:
        self._policy.reset()


class DroidExteriorImageKeyAdapter(base_policy.BasePolicy):
    """Map a deployment camera key to the key used during model training.

    DROID Infra numbers its runtime exterior camera slots differently from the
    wipe-board LeRobot dataset. This adapter makes that boundary explicit while
    leaving the model's training data transform unchanged.
    """

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        *,
        deployment_image_key: str,
        training_image_key: str,
    ) -> None:
        self._policy = policy
        self._deployment_image_key = deployment_image_key
        self._training_image_key = training_image_key

    @override
    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        if self._deployment_image_key not in obs:
            raise KeyError(
                f"Observation is missing {self._deployment_image_key!r}, required "
                f"for the trained camera slot {self._training_image_key!r}."
            )
        remapped_obs = dict(obs)
        remapped_obs[self._training_image_key] = obs[self._deployment_image_key]
        return self._policy.infer(remapped_obs)

    @override
    def reset(self) -> None:
        self._policy.reset()
