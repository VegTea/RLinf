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

"""Deterministic action-chunk evaluation shared by OpenPI SFT workers."""

from collections.abc import Mapping, Sequence

import numpy as np


def select_uniform_chunk_indices(
    episode_lengths: Mapping[int, int],
    episodes: Sequence[int],
    action_horizon: int,
    chunks_per_episode: int,
) -> tuple[list[int], dict[int, list[int]]]:
    """Select local indices whose action chunks do not use episode-end padding."""
    if action_horizon <= 0 or chunks_per_episode <= 0:
        raise ValueError("action_horizon and chunks_per_episode must be positive.")

    dataset_indices: list[int] = []
    offsets_by_episode: dict[int, list[int]] = {}
    dataset_offset = 0
    for episode in sorted(int(index) for index in episodes):
        episode_length = int(episode_lengths[episode])
        valid_starts = episode_length - action_horizon + 1
        if valid_starts < chunks_per_episode:
            raise ValueError(
                f"Episode {episode} has only {valid_starts} valid chunk starts, "
                f"cannot select {chunks_per_episode}."
            )
        offsets = (
            np.rint(np.linspace(0, valid_starts - 1, chunks_per_episode))
            .astype(int)
            .tolist()
        )
        if len(set(offsets)) != chunks_per_episode:
            raise RuntimeError(
                f"Uniform chunk selection produced duplicate offsets for episode {episode}."
            )
        offsets_by_episode[episode] = offsets
        dataset_indices.extend(dataset_offset + offset for offset in offsets)
        dataset_offset += episode_length
    return dataset_indices, offsets_by_episode


def make_fixed_eval_noise(
    seed: int, num_samples: int, action_horizon: int, action_dim: int
) -> np.ndarray:
    """Create noise numerically identical to the OpenPI JAX evaluator."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (num_samples, action_horizon, action_dim), dtype=np.float32
    )


class ActionMetricAccumulator:
    """Accumulate element-weighted decoded action errors."""

    _NAMES = (
        "action_normalized",
        "joint_velocity",
        "gripper_position",
        "joint_position",
    )

    def __init__(self, control_frequency_hz: float):
        if control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive.")
        self._frequency = control_frequency_hz
        self._squared_sums = dict.fromkeys(self._NAMES, 0.0)
        self._absolute_sums = dict.fromkeys(self._NAMES, 0.0)
        self._counts = dict.fromkeys(self._NAMES, 0)
        self.num_chunks = 0

    def _add(self, name: str, prediction: np.ndarray, target: np.ndarray) -> None:
        error = np.asarray(prediction, dtype=np.float64) - np.asarray(
            target, dtype=np.float64
        )
        self._squared_sums[name] += float(np.square(error).sum())
        self._absolute_sums[name] += float(np.abs(error).sum())
        self._counts[name] += error.size

    def update(
        self,
        predicted_normalized: np.ndarray,
        expert_normalized: np.ndarray,
        predicted_native: np.ndarray,
        expert_native: np.ndarray,
        current_joint_position: np.ndarray,
    ) -> None:
        """Update normalized, velocity, gripper, and integrated-position errors."""
        predicted_normalized = np.asarray(predicted_normalized)[..., :8]
        expert_normalized = np.asarray(expert_normalized)[..., :8]
        predicted_native = np.asarray(predicted_native)[..., :8]
        expert_native = np.asarray(expert_native)[..., :8]
        current_joint_position = np.asarray(current_joint_position)[..., :7]

        self._add("action_normalized", predicted_normalized, expert_normalized)
        self._add("joint_velocity", predicted_native[..., :7], expert_native[..., :7])
        self._add(
            "gripper_position", predicted_native[..., 7:8], expert_native[..., 7:8]
        )
        predicted_position = current_joint_position[:, None, :] + np.cumsum(
            predicted_native[..., :7] / self._frequency, axis=1
        )
        expert_position = current_joint_position[:, None, :] + np.cumsum(
            expert_native[..., :7] / self._frequency, axis=1
        )
        self._add("joint_position", predicted_position, expert_position)
        self.num_chunks += predicted_normalized.shape[0]

    def as_totals(self) -> np.ndarray:
        """Serialize sums/counts for a distributed SUM reduction."""
        values: list[float] = []
        for name in self._NAMES:
            values.extend(
                [
                    self._squared_sums[name],
                    self._absolute_sums[name],
                    float(self._counts[name]),
                ]
            )
        values.append(float(self.num_chunks))
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def metrics_from_totals(cls, totals: np.ndarray) -> dict[str, float]:
        """Convert globally reduced totals into final mean metrics."""
        totals = np.asarray(totals, dtype=np.float64)
        expected_size = len(cls._NAMES) * 3 + 1
        if totals.shape != (expected_size,):
            raise ValueError(
                f"Expected {expected_size} reduced totals, got {totals.shape}."
            )
        metrics: dict[str, float] = {}
        for index, name in enumerate(cls._NAMES):
            squared_sum, absolute_sum, count = totals[index * 3 : index * 3 + 3]
            if count <= 0:
                raise RuntimeError(f"No elements accumulated for {name}.")
            metrics[f"{name}_mse"] = float(squared_sum / count)
            metrics[f"{name}_mae"] = float(absolute_sum / count)
        metrics["num_chunks"] = float(totals[-1])
        return metrics
