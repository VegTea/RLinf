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

"""Tests for pi05_droid expert-chunk server evaluation helpers."""

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "toolkits"
    / "standalone_eval_scripts"
    / "openpi"
    / "evaluate_pi05_droid_server.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_pi05_droid_server", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

absolute_chunk_to_droid_actions = _MODULE.absolute_chunk_to_droid_actions
compute_chunk_metrics = _MODULE.compute_chunk_metrics
select_uniform_chunk_starts = _MODULE.select_uniform_chunk_starts
resolve_exterior_camera = _MODULE.resolve_exterior_camera
make_observation = _MODULE._make_observation


def test_select_uniform_chunk_starts_avoids_end_padding():
    starts = select_uniform_chunk_starts(
        100,
        action_horizon=15,
        num_samples=10,
    )

    assert len(starts) == 10
    assert starts[0] == 0
    assert starts[-1] == 85
    assert np.all(np.diff(starts) > 0)


def test_absolute_chunk_to_droid_actions_uses_previous_target():
    current = np.zeros(7)
    positions = np.stack([np.full(7, 0.1), np.full(7, 0.3)])
    absolute = np.concatenate([positions, np.array([[0.2], [0.8]])], axis=1)

    droid = absolute_chunk_to_droid_actions(
        current,
        absolute,
        control_frequency_hz=10.0,
    )

    np.testing.assert_allclose(droid[:, :7], [[1.0] * 7, [2.0] * 7])
    np.testing.assert_allclose(droid[:, 7], [0.2, 0.8])


def test_compute_chunk_metrics_reports_zero_for_exact_match():
    actions = np.arange(2 * 3 * 8, dtype=np.float64).reshape(2, 3, 8) / 100
    current = np.zeros((2, 7), dtype=np.float64)

    metrics = compute_chunk_metrics(
        actions,
        actions.copy(),
        current,
        control_frequency_hz=15.0,
    )

    assert all(value == 0.0 for value in metrics["aggregate"].values())
    assert all(
        value == 0.0 for chunk in metrics["per_chunk"] for value in chunk.values()
    )


def test_resolve_exterior_camera_uses_server_metadata():
    camera, image_key = resolve_exterior_camera(
        "auto",
        {
            "exterior_camera": "left",
            "exterior_image_key": "observation/exterior_image_0_left",
        },
    )

    assert camera == "left"
    assert image_key == "observation/exterior_image_0_left"


def test_resolve_exterior_camera_rejects_mismatch():
    import pytest

    with pytest.raises(ValueError, match="server expects"):
        resolve_exterior_camera(
            "right",
            {
                "exterior_camera": "left",
                "exterior_image_key": "observation/exterior_image_0_left",
            },
        )


def test_dataset_evaluation_uses_training_image_but_deployment_key():
    left_image = np.full((2, 3, 3), 10, dtype=np.uint8)
    right_image = np.full((2, 3, 3), 20, dtype=np.uint8)
    sample = {
        "exterior_image_1_left": left_image,
        "exterior_image_2_left": right_image,
        "wrist_image_left": np.zeros((2, 3, 3), dtype=np.uint8),
        "joint_position": np.zeros(7, dtype=np.float32),
        "gripper_position": np.zeros(1, dtype=np.float32),
        "task": "wipe the whiteboard",
    }

    observation = make_observation(sample, None, "right")

    assert "observation/exterior_image_1_left" in observation
    np.testing.assert_array_equal(
        observation["observation/exterior_image_1_left"], right_image
    )
