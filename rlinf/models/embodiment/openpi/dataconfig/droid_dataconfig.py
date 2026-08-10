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

"""OpenPI data configuration for the official DROID joint-velocity policy."""

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert a LeRobot image to HWC uint8, as required by OpenPI transforms."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.ascontiguousarray(image.transpose(1, 2, 0))
    return image


def transform_pi05_droid_numeric_episode(
    joint_positions: np.ndarray,
    gripper_positions: np.ndarray,
    absolute_actions: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
    control_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one numeric DROID episode exactly like the OpenPI loader."""
    states = np.concatenate([joint_positions, gripper_positions], axis=-1)
    frame_offsets = np.arange(action_horizon)[None, :]
    action_indices = np.minimum(
        np.arange(len(states))[:, None] + frame_offsets, len(states) - 1
    )
    action_chunks = absolute_actions[action_indices].copy()
    previous_joint_positions = np.concatenate(
        [joint_positions[:, None, :], action_chunks[:, :-1, :7]], axis=1
    )
    action_chunks[:, :, :7] = (
        action_chunks[:, :, :7] - previous_joint_positions
    ) * control_frequency_hz
    return (
        _transforms.pad_to_dim(states, action_dim),
        _transforms.pad_to_dim(action_chunks, action_dim),
    )


@dataclasses.dataclass(frozen=True)
class JointPositionActionsToVelocity(_transforms.DataTransformFn):
    """Convert absolute Panda joint targets to DROID joint-velocity actions.

    The official ``pi05_droid`` expert is trained with seven joint velocities and
    an absolute gripper command. RLinf-collected Franka datasets commonly store
    absolute joint targets instead, so every action chunk is differenced from the
    current state / preceding target and scaled by the data collection frequency.
    """

    control_frequency_hz: float

    def __call__(self, data: dict) -> dict:
        # The same input transform is used for training and policy inference.
        # Inference observations intentionally do not contain expert actions.
        if "actions" not in data:
            return data

        actions = np.asarray(data["actions"]).copy()
        state = np.asarray(data["state"])
        if actions.ndim != 2 or actions.shape[-1] < 8:
            raise ValueError(
                "Expected an action chunk shaped (horizon, >=8) containing "
                "seven joint positions and one gripper position, got "
                f"{actions.shape}."
            )
        if state.ndim != 1 or state.shape[0] < 7:
            raise ValueError(
                "Expected a state vector with at least seven joint positions, got "
                f"{state.shape}."
            )

        previous_positions = np.concatenate(
            [state[np.newaxis, :7], actions[:-1, :7]], axis=0
        )
        actions[:, :7] = (
            actions[:, :7] - previous_positions
        ) * self.control_frequency_hz
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen=True)
class DroidJointVelocityInputs(_transforms.DataTransformFn):
    """Map the local DROID-style LeRobot fields to official OpenPI inputs."""

    action_dim: int
    exterior_image_key: str = "observation/exterior_image_2_left"
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        if "observation/state" in data:
            state = np.asarray(data["observation/state"])
        else:
            state = np.concatenate(
                [
                    np.asarray(data["observation/joint_position"]),
                    np.asarray(data["observation/gripper_position"]).reshape(-1),
                ]
            )
        if state.shape != (8,):
            raise ValueError(
                "Official pi05_droid expects a 7D Panda joint state plus one "
                f"gripper state, got {state.shape}."
            )

        base_image = _parse_image(data[self.exterior_image_key])
        wrist_image = _parse_image(data["observation/wrist_image_left"])
        # Map the selected exterior view and wrist camera to the two active
        # pi0.5-DROID image slots. The wipe-board default uses the right view.
        # Keep the third pi0.5 image slot masked rather than assigning the
        # other exterior camera to a pretrained wrist-camera slot.
        images = (base_image, wrist_image, np.zeros_like(base_image))
        inputs = {
            "state": _transforms.pad_to_dim(state, self.action_dim),
            "image": dict(
                zip(
                    ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
                    images,
                    strict=True,
                )
            ),
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = _transforms.pad_to_dim(
                np.asarray(data["actions"]), self.action_dim
            )
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode() if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class DroidJointVelocityOutputs(_transforms.DataTransformFn):
    """Expose DROID's seven joint velocities and absolute gripper command."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :8]}


@dataclasses.dataclass(frozen=True)
class LeRobotDroidJointVelocityDataConfig(DataConfigFactory):
    """LeRobot configuration compatible with the official ``pi05_droid`` model."""

    control_frequency_hz: float = 15.0
    default_prompt: str | None = None
    exterior_image_key: str = "observation/exterior_image_2_left"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                DroidJointVelocityInputs(
                    action_dim=model_config.action_dim,
                    exterior_image_key=self.exterior_image_key,
                    model_type=model_config.model_type,
                ),
                JointPositionActionsToVelocity(self.control_frequency_hz),
            ],
            outputs=[DroidJointVelocityOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
