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

"""Serve pi05_droid with wipe-board inputs and absolute joint outputs."""

import logging
import pathlib

import openpi.policies.policy_config as policy_config
import openpi.shared.normalize as normalize
import tyro
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.droid_deployment_policy import (
    DroidAbsoluteJointPositionPolicy,
)


def create_pi05_droid_deployment_policy(
    checkpoint_dir: pathlib.Path,
    norm_stats_dir: pathlib.Path,
    *,
    control_frequency_hz: float,
    pytorch_device: str,
    default_prompt: str | None,
) -> DroidAbsoluteJointPositionPolicy:
    """Load a local pi05_droid checkpoint for wipe-board deployment."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    norm_stats_dir = norm_stats_dir.expanduser().resolve()
    if not (checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"PyTorch checkpoint not found at {checkpoint_dir / 'model.safetensors'}."
        )
    if not (norm_stats_dir / "norm_stats.json").is_file():
        raise FileNotFoundError(
            f"Normalization statistics not found at "
            f"{norm_stats_dir / 'norm_stats.json'}."
        )

    train_config = get_openpi_config(
        "pi05_droid",
        model_path=str(checkpoint_dir),
        data_kwargs={
            "control_frequency_hz": control_frequency_hz,
            "exterior_image_key": "observation/exterior_image_2_left",
            "norm_stats_path": str(norm_stats_dir),
        },
    )
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt,
        norm_stats=normalize.load(norm_stats_dir),
        pytorch_device=pytorch_device,
    )
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
    """Start the official OpenPI WebSocket server for pi05_droid.

    The server expects the same right-exterior and wrist camera keys used by
    wipe-board SFT. Its response contains absolute joint-position chunks so a
    DROID robot client does not need to integrate model velocities itself.
    """
    logging.basicConfig(level=logging.INFO)
    policy = create_pi05_droid_deployment_policy(
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

    logging.info("Serving pi05_droid on ws://%s:%d with %s", host, port, metadata)
    WebsocketPolicyServer(
        policy, host=host, port=port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
