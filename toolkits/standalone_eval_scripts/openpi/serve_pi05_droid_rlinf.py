#!/usr/bin/env python3
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

"""Serve an RLinf openpi_pytorch pi05_droid checkpoint via WebSocket.

Key difference from ``serve_pi05_droid.py``: this loads the RLinf vendored
``Pi0`` model (keys ``img.*``, ``llm.*``) instead of the upstream
``PI0Pytorch`` (keys ``paligemma_with_expert.*``).

Usage:
  PYTHONPATH="${REPO_ROOT}" .venv/bin/python \\
    toolkits/standalone_eval_scripts/openpi/serve_pi05_droid_rlinf.py \\
    --checkpoint-dir checkpoints/pi05_droid_openpi_rlinf_best_step10000_inference \\
    --norm-stats-dir checkpoints/pi05_droid_openpi_rlinf_best_step10000_inference/wipe_board_v1_zed196_force
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import safetensors.torch
import torch
import tyro
from openpi.policies import policy as _policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize as _normalize
import openpi.transforms as _transforms

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.droid_deployment_policy import (
    DroidAbsoluteJointPositionPolicy,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config


def _create_rlinf_pi05_droid_policy(
    checkpoint_dir: pathlib.Path,
    norm_stats_dir: pathlib.Path,
    *,
    control_frequency_hz: float,
    pytorch_device: str,
    default_prompt: str | None,
) -> DroidAbsoluteJointPositionPolicy:
    """Load an RLinf openpi_pytorch checkpoint and build a serving policy."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    norm_stats_dir = norm_stats_dir.expanduser().resolve()

    # ── validate required files ──────────────────────────────────────
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not (norm_stats_dir / "norm_stats.json").is_file():
        raise FileNotFoundError(
            f"norm_stats.json not found: {norm_stats_dir / 'norm_stats.json'}"
        )

    # ── 1. build transforms pipeline via the upstream openpi config ──
    train_config = get_openpi_config(
        "pi05_droid",
        model_path=str(checkpoint_dir),
        data_kwargs={
            "control_frequency_hz": control_frequency_hz,
            "exterior_image_key": "observation/exterior_image_2_left",
            "norm_stats_path": str(norm_stats_dir),
        },
    )
    data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model
    )
    norm_stats = _normalize.load(norm_stats_dir)

    # ── 2. load the RLinf openpi_pytorch Pi0 model ───────────────────
    pi0_config = Pi0Config(
        pi05=True,
        action_horizon=15,
        action_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
        max_token_len=200,
        pcd=False,
    )
    pi0_model = pi0_config.create()
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    pi0_model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in pi0_model.parameters())
    logging.info("Loaded RLinf Pi0 model (%.2fB params) from %s", n_params / 1e9, weights_path)

    # ── 3. monkey-patch sample_actions to match upstream convention ──
    # upstream PI0Pytorch:  sample_actions(self, device, observation, noise=..., num_steps=...)
    # RLinf Pi0:            sample_actions(self, observation, *, num_steps=..., noise=..., rng=...)
    _original_sample_actions = pi0_model.sample_actions

    def _adapted_sample_actions(
        device: str,
        observation: Any,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        **kwargs: Any,
    ) -> torch.Tensor:
        del device, kwargs
        return _original_sample_actions(
            observation, num_steps=num_steps, noise=noise
        )

    pi0_model.sample_actions = _adapted_sample_actions  # type: ignore[method-assign]

    # ── 4. wrap in openpi Policy (handles transforms + inference) ────
    policy = _policy.Policy(
        pi0_model,
        transforms=[
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
        ],
        is_pytorch=True,
        pytorch_device=pytorch_device,
    )

    # ── 5. wrap velocity policy as absolute-joint-position policy ────
    return DroidAbsoluteJointPositionPolicy(
        policy,
        control_frequency_hz=control_frequency_hz,
    )


def main(
    checkpoint_dir: pathlib.Path,
    norm_stats_dir: pathlib.Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    control_frequency_hz: float = 15.0,
    pytorch_device: str = "cuda",
    default_prompt: str | None = None,
    dry_run: bool = False,
) -> None:
    """Start the OpenPI WebSocket server for an RLinf pi05_droid checkpoint."""
    logging.basicConfig(level=logging.INFO)

    policy = _create_rlinf_pi05_droid_policy(
        checkpoint_dir,
        norm_stats_dir,
        control_frequency_hz=control_frequency_hz,
        pytorch_device=pytorch_device,
        default_prompt=default_prompt,
    )

    metadata = {
        "action_space": "joint_position",
        "native_model_action_space": "joint_velocity",
        "action_horizon": 15,
        "control_frequency_hz": control_frequency_hz,
        "exterior_image_key": "observation/exterior_image_2_left",
        "wrist_image_key": "observation/wrist_image_left",
    }
    if dry_run:
        logging.info("Policy loaded successfully; dry run complete: %s", metadata)
        return

    logging.info(
        "Serving pi05_droid (RLinf openpi_pytorch) on ws://%s:%d with %s",
        host,
        port,
        metadata,
    )
    WebsocketPolicyServer(
        policy, host=host, port=port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
