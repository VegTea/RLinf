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

"""Serve a RECAP/CFG OpenPI policy through OpenPI's websocket protocol.

This script keeps the OpenPI websocket/msgpack communication protocol intact and
only swaps the in-process policy implementation to RLinf's CFG policy. Existing
OpenPI websocket clients can keep sending the same observations; the server
adds the positive/negative guidance prompts internally and forces positive CFG
guidance during action sampling.
"""

from __future__ import annotations

import argparse
import copy
import logging
import pathlib
import sys
import time
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import numpy as np
import openpi.transforms as transforms
import safetensors.torch
import torch
from openpi.policies import policy as _policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import checkpoints as _checkpoints

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation,
    OpenPi0Config,
    OpenPi0ForCFGActionPrediction,
)


logger = logging.getLogger(__name__)


def _as_batch(value: Any) -> Any:
    """Add a batch dimension to array leaves while preserving strings."""
    if isinstance(value, str):
        return np.asarray([value], dtype=object)
    if isinstance(value, np.ndarray):
        return value[None, ...]
    if torch.is_tensor(value):
        return value[None, ...]
    return value


def _to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device).contiguous()
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _extract_prompt(obs: dict[str, Any]) -> str:
    """Return the task prompt from common OpenPI request shapes."""
    if "task" in obs:
        return str(obs["task"])
    if "prompt" in obs:
        return str(obs["prompt"])
    observation = obs.get("observation")
    if isinstance(observation, dict):
        if "task" in observation:
            return str(observation["task"])
        if "prompt" in observation:
            return str(observation["prompt"])
    raise KeyError(
        "Cannot find task prompt in request. Expected key 'task' or 'prompt'."
    )


def _with_lerobot_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Mirror OpenPI websocket requests into the LeRobot shape expected by RECAP."""
    images = obs.get("images") or {}
    if "observation" not in obs and ("images" in obs or "state" in obs):
        obs["observation"] = {"images": images, "state": obs.get("state")}
    for key, value in images.items():
        obs.setdefault(f"observation.images.{key}", value)
    if "state" in obs:
        obs.setdefault("observation.state", obs["state"])
    return obs


class RecapCfgPolicy(_policy.BasePolicy):
    """OpenPI protocol-compatible policy that runs RLinf RECAP/CFG inference."""

    def __init__(
        self,
        model: OpenPi0ForCFGActionPrediction,
        *,
        device: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._metadata = metadata or {}

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = copy.deepcopy(obs)
        prompt = _extract_prompt(inputs)
        inputs.setdefault("task", prompt)
        inputs["prompt"] = prompt
        inputs = _with_lerobot_observation(inputs)
        inputs.setdefault(
            "action",
            np.zeros(
                (
                    int(self._metadata.get("action_horizon", 50)),
                    int(self._metadata.get("action_dim", 14)),
                ),
                dtype=np.float32,
            ),
        )
        inputs["positive_guidance_prompt"] = f"{prompt}\nAdvantage: positive"
        inputs["negative_guidance_prompt"] = f"{prompt}\nAdvantage: negative"

        batched = jax.tree.map(_as_batch, inputs)
        processed_obs = self._model.input_transform(batched, transpose=False)
        processed_obs = _to_device(processed_obs, self._device)
        observation = Observation.from_dict(processed_obs)

        sample_kwargs = {}
        if noise is not None:
            noise_tensor = torch.from_numpy(noise).to(self._device)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor[None, ...]
            sample_kwargs["noise"] = noise_tensor

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = self._model.sample_actions(observation, **sample_kwargs)
        model_time = time.monotonic() - start_time

        transformed = self._model.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )
        result = jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu())
            if torch.is_tensor(x)
            else x,
            transformed,
        )
        result["policy_timing"] = {"infer_ms": model_time * 1000}
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def create_recap_cfg_policy(args: argparse.Namespace) -> RecapCfgPolicy:
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {model_path}")

    train_config = get_openpi_config(
        args.config_name,
        model_path=str(checkpoint_dir),
        data_kwargs={"repo_id": args.repo_id},
    )
    model_config = OpenPi0Config(**train_config.model.__dict__)
    model_config = dataclass_replace(
        model_config,
        config_name=args.config_name,
        num_images_in_input=args.num_images_in_input,
        action_chunk=args.action_chunk,
        action_env_dim=args.action_env_dim,
        cfgrl_guidance_scale=args.guidance_scale,
        guidance_type=args.guidance_type,
        positive_only_conditional=args.positive_only_conditional,
        train_expert_only=False,
    )

    model = OpenPi0ForCFGActionPrediction(model_config)
    safetensors.torch.load_model(model, str(model_path), strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    model.setup_wrappers(
        transforms=[
            *data_config.repack_transforms.inputs,
            transforms.InjectDefaultPrompt(args.default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *data_config.repack_transforms.outputs,
        ],
    )

    return RecapCfgPolicy(
        model,
        device=args.device,
        metadata={
            "protocol_version": "1.0",
            "policy_name": "RecapCfgPolicy",
            "control_mode": "joints",
            "action_horizon": args.action_chunk,
            "action_dim": args.action_env_dim,
            "state_dim": args.action_env_dim,
            "image_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "image_shape": [3, 480, 640],
            "expects_prompt": True,
            "accepts_compressed_images": True,
            "extra": {"model_action_dim": getattr(model_config, "action_dim", None)},
            **(train_config.policy_metadata or {}),
            "checkpoint_dir": str(checkpoint_dir),
            "config_name": args.config_name,
            "repo_id": args.repo_id,
            "guidance_type": args.guidance_type,
            "guidance_scale": args.guidance_scale,
        },
    )


def dataclass_replace(model_config: OpenPi0Config, **kwargs: Any) -> OpenPi0Config:
    # Avoid importing dataclasses at module import time before OpenPI finishes
    # registering config classes in some environments.
    import dataclasses

    return dataclasses.replace(model_config, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config-name", default="pi05_yam")
    parser.add_argument(
        "--repo-id",
        default="assets/tower-of-hanoi-game/expert-success-hil-suffix-mix-data",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device for the policy, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--num-images-in-input", type=int, default=3)
    parser.add_argument("--action-chunk", type=int, default=50)
    parser.add_argument("--action-env-dim", type=int, default=14)
    parser.add_argument("--guidance-type", default="positive")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--positive-only-conditional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--default-prompt", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    policy = create_recap_cfg_policy(args)
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    logger.info("Serving RECAP/CFG policy on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
