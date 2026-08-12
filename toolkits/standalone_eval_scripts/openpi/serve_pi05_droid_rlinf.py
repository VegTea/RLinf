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

import json
import logging
import pathlib
from typing import Any

import numpy as np
import openpi.transforms as _transforms
import safetensors.torch
import torch
import tyro
from openpi.policies import policy as _policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize as _normalize

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.droid_deployment_policy import (
    DroidAbsoluteJointPositionPolicy,
)
from rlinf.models.embodiment.openpi.policies.observation_recording_policy import (
    DroidObservationRecordingPolicy,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as _rlinf_model
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

_EXTERIOR_CAMERA_KEY = {
    "left": "observation/exterior_image_1_left",
    "right": "observation/exterior_image_2_left",
}


def _adapt_sample_actions_for_openpi(pi0_model: Any) -> None:
    """Adapt RLinf's sampler signature and ensure inference never records grads."""
    original_sample_actions = pi0_model.sample_actions

    @torch.inference_mode()
    def adapted_sample_actions(
        device: str,
        observation: Any,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        **kwargs: Any,
    ) -> torch.Tensor:
        del device, kwargs
        images = {}
        for name, image in observation.images.items():
            if image.ndim != 4:
                raise ValueError(
                    f"Expected batched image {name!r} with four dimensions, "
                    f"got {tuple(image.shape)}."
                )
            # Upstream OpenPI's PyTorch Observation converts uint8 BHWC inputs
            # to normalized BCHW. RLinf's vendored Pi0 expects normalized BHWC.
            if image.shape[1] == 3:
                image = image.permute(0, 2, 3, 1).contiguous()
            elif image.shape[-1] != 3:
                raise ValueError(
                    f"Expected RGB image {name!r}, got {tuple(image.shape)}."
                )
            images[name] = image
        observation = _rlinf_model.Observation(
            images=images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=getattr(observation, "token_ar_mask", None),
            token_loss_mask=getattr(observation, "token_loss_mask", None),
            pcd_xyz=getattr(observation, "pcd_xyz", None),
        )
        return original_sample_actions(
            observation,
            num_steps=num_steps,
            noise=noise,
        )

    pi0_model.sample_actions = adapted_sample_actions


def _create_rlinf_pi05_droid_policy(
    checkpoint_dir: pathlib.Path,
    norm_stats_dir: pathlib.Path,
    *,
    control_frequency_hz: float,
    pytorch_device: str,
    default_prompt: str | None,
    exterior_camera: str = "right",
) -> tuple[DroidAbsoluteJointPositionPolicy, str]:
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
    if exterior_camera not in _EXTERIOR_CAMERA_KEY:
        raise ValueError(
            f"exterior_camera must be one of {list(_EXTERIOR_CAMERA_KEY)}, "
            f"got {exterior_camera!r}."
        )
    if not np.isfinite(control_frequency_hz) or control_frequency_hz <= 0:
        raise ValueError(
            "control_frequency_hz must be finite and positive, got "
            f"{control_frequency_hz}."
        )
    image_key = _EXTERIOR_CAMERA_KEY[exterior_camera]

    # ── 1. build transforms pipeline via the upstream openpi config ──
    train_config = get_openpi_config(
        "pi05_droid",
        model_path=str(checkpoint_dir),
        data_kwargs={
            "control_frequency_hz": control_frequency_hz,
            "exterior_image_key": image_key,
            "norm_stats_path": str(norm_stats_dir),
        },
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _normalize.load(norm_stats_dir)

    # ── 2. load the RLinf openpi_pytorch Pi0 model ───────────────────
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_fields = {
        "pi05",
        "action_horizon",
        "action_dim",
        "paligemma_variant",
        "action_expert_variant",
        "dtype",
        "max_token_len",
        "pcd",
        "discrete_state_input",
    }
    pi0_config = Pi0Config(
        **{key: value for key, value in config_data.items() if key in config_fields}
    )
    if not pi0_config.pi05:
        raise ValueError(f"Expected a pi0.5 checkpoint, got {config_path}.")
    if pi0_config.action_horizon != 15 or pi0_config.action_dim != 32:
        raise ValueError(
            "pi05_droid deployment requires action_horizon=15 and action_dim=32, "
            f"got {pi0_config.action_horizon} and {pi0_config.action_dim}."
        )
    if pi0_config.dtype != "bfloat16":
        raise ValueError(
            f"Only bfloat16 RLinf deployment is supported, got {pi0_config.dtype!r}."
        )
    pi0_model = pi0_config.create()
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    pi0_model.load_state_dict(state_dict, strict=True)
    pi0_model = pi0_model.to(dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in pi0_model.parameters())
    logging.info(
        "Loaded RLinf Pi0 model (%.2fB params, %s) from %s",
        n_params / 1e9,
        pi0_config.dtype,
        weights_path,
    )

    # ── 3. monkey-patch sample_actions to match upstream convention ──
    # upstream PI0Pytorch:  sample_actions(self, device, observation, noise=..., num_steps=...)
    # RLinf Pi0:            sample_actions(self, observation, *, num_steps=..., noise=..., rng=...)
    _adapt_sample_actions_for_openpi(pi0_model)

    # ── 4. wrap in openpi Policy (handles transforms + inference) ────
    policy = _policy.Policy(
        pi0_model,
        transforms=[
            _transforms.InjectDefaultPrompt(default_prompt),
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
    ), image_key


def main(
    checkpoint_dir: pathlib.Path,
    norm_stats_dir: pathlib.Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    control_frequency_hz: float = 15.0,
    pytorch_device: str = "cuda",
    default_prompt: str | None = None,
    dry_run: bool = False,
    exterior_camera: str = "right",
    observation_record_dir: pathlib.Path | None = None,
    observation_record_fps: float = 15.0,
) -> None:
    """Start the OpenPI WebSocket server for an RLinf pi05_droid checkpoint.

    Args:
        exterior_camera: Which DROID exterior camera to use as the main view.
            ``"left"`` maps to ``observation/exterior_image_1_left``;
            ``"right"`` maps to ``observation/exterior_image_2_left``.
        observation_record_dir: Optional root under which raw observations and
            H.264 input videos are recorded in a timestamped session directory.
        observation_record_fps: Frame rate written into observation videos.
    """
    logging.basicConfig(level=logging.INFO)

    policy, image_key = _create_rlinf_pi05_droid_policy(
        checkpoint_dir,
        norm_stats_dir,
        control_frequency_hz=control_frequency_hz,
        pytorch_device=pytorch_device,
        default_prompt=default_prompt,
        exterior_camera=exterior_camera,
    )

    metadata = {
        "action_space": "joint_position",
        "native_model_action_space": "joint_velocity",
        "action_horizon": 15,
        "control_frequency_hz": control_frequency_hz,
        "exterior_image_key": image_key,
        "wrist_image_key": "observation/wrist_image_left",
    }
    recorder = None
    if observation_record_dir is not None:
        recorder = DroidObservationRecordingPolicy(
            policy,
            output_root=observation_record_dir,
            exterior_image_key=image_key,
            fps=observation_record_fps,
        )
        policy = recorder
        metadata["observation_record_dir"] = str(recorder.output_dir)
    if dry_run:
        logging.info("Policy loaded successfully; dry run complete: %s", metadata)
        return

    logging.info(
        "Serving pi05_droid (RLinf openpi_pytorch) on ws://%s:%d with %s",
        host,
        port,
        metadata,
    )
    try:
        WebsocketPolicyServer(
            policy, host=host, port=port, metadata=metadata
        ).serve_forever()
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    tyro.cli(main)
