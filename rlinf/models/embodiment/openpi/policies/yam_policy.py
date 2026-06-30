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

"""OpenPI transforms for dual-arm YAM datasets."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        numerator = horn_radius**2 + linear_position**2 - arm_length**2
        denominator = 2 * horn_radius * linear_position
        value = numerator / denominator
        return np.arcsin(np.clip(value, -1.0, 1.0))

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    value = _unnormalize(value, min_val=0.4, max_val=1.5)
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _yam_joint_flip_mask() -> np.ndarray:
    """Return the YAM-to-pi joint-angle sign mask."""
    return np.array([1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])


def _decode_yam_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        state = _yam_joint_flip_mask() * state[: _yam_joint_flip_mask().shape[0]]
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_yam_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _yam_joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_yam_actions_inv(
    actions: np.ndarray, *, adapt_to_pi: bool = False
) -> np.ndarray:
    if adapt_to_pi:
        actions = _yam_joint_flip_mask() * actions[:, : len(_yam_joint_flip_mask())]
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions


def _decode_yam(data: dict, *, adapt_to_pi: bool = False) -> dict:
    state = np.asarray(data["state"])
    state = _decode_yam_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            return einops.rearrange(img, "c h w -> h w c")
        return img

    images = data["images"]
    data["images"] = {name: convert_image(img) for name, img in images.items()}
    data["state"] = state
    return data


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Inputs for dual-arm YAM policy training and inference.

    Expected input keys:
    - ``images``: dict with ``cam_high``, ``cam_left_wrist``,
      ``cam_right_wrist`` images.
    - ``state``: 14-D vector ``[L joints 1-6, L gripper, R joints 1-6, R gripper]``.
    - ``actions``: optional ``[action_horizon, 14]`` action chunk.
    """

    action_dim: int
    adapt_to_pi: bool = True
    model_type: _model.ModelType = _model.ModelType.PI0
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_yam(data, adapt_to_pi=self.adapt_to_pi)
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        missing_cameras = set(self.EXPECTED_CAMERAS) - set(in_images)
        if missing_cameras:
            raise ValueError(
                f"Expected images to contain {self.EXPECTED_CAMERAS}, "
                f"got {tuple(in_images)}; missing {tuple(sorted(missing_cameras))}"
            )

        base_image = in_images["cam_high"]
        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            images = {"base_0_rgb": base_image}
            image_masks = {"base_0_rgb": np.True_}
            extra_image_names = {
                "left_wrist_0_rgb": "cam_left_wrist",
                "right_wrist_0_rgb": "cam_right_wrist",
            }
        elif self.model_type == _model.ModelType.PI0_FAST:
            images = {"base_0_rgb": base_image}
            image_masks = {"base_0_rgb": np.True_}
            extra_image_names = {
                "base_1_rgb": "cam_left_wrist",
                "wrist_0_rgb": "cam_right_wrist",
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_yam_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Outputs for dual-arm YAM policy inference."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_yam_actions(actions, adapt_to_pi=self.adapt_to_pi)}
