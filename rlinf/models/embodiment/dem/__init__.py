# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only DEM integration. Training uses the upstream MeanFlow objective."""

import hashlib
import json
from pathlib import Path

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.dem.stack_cube_io import (
    INSTRUCTION,
    STATE_ROTATION_ENCODINGS,
    environment_state_to_dataset,
    prepare_stack_actions,
)


class DEMPolicy(torch.nn.Module, BasePolicy):
    """Translate RLinf Stack-Cube observations to the deployed DEM protocol."""

    def __init__(
        self,
        policy: torch.nn.Module,
        execute_steps: int,
        seed: int,
        state_rotation_encoding: str,
    ):
        super().__init__()
        if state_rotation_encoding not in STATE_ROTATION_ENCODINGS:
            raise ValueError("Explicit supported state_rotation_encoding is required")
        self.state_rotation_encoding = state_rotation_encoding
        self.policy = policy
        self.execute_steps = execute_steps
        self.seed = seed
        self._generator = None
        if not 0 < execute_steps <= policy.config.chunk:
            raise ValueError("DEM execute_steps exceeds prediction horizon")

    def default_forward(self, **kwargs):
        raise NotImplementedError("DEM RL actor training is not implemented; use SFT")

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "eval", **kwargs):
        """Return controller-space actions [B,K,7] and no fictitious RL outputs."""
        from dem import preprocess_images

        if mode != "eval":
            raise NotImplementedError("DEM is currently supported only in evaluation")
        texts = env_obs["task_descriptions"]
        if any(text != INSTRUCTION for text in texts):
            raise ValueError(
                "Checkpoint is trained only for the Stack-Cube instruction"
            )
        if "eef_quat_xyzw" not in env_obs:
            raise ValueError(
                "DEM requires native eef_quat_xyzw; enable env.eval.include_native_eef_quat"
            )
        device = self.policy.device
        states = environment_state_to_dataset(
            env_obs["states"],
            env_obs["eef_quat_xyzw"],
            encoding=self.state_rotation_encoding,
        ).to(device)
        images = torch.stack([env_obs["main_images"], env_obs["wrist_images"]], dim=1)
        images = preprocess_images(images, size=256, device=device)
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device).manual_seed(self.seed)
        noise = torch.randn(
            (states.shape[0], self.policy.config.chunk, 7),
            device=device,
            generator=self._generator,
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            vision = self.policy.encode_vision(images)
        # DEM.act unnormalizes exactly once, and samples the entire horizon.
        actions = self.policy.act(
            None, states, texts=texts, vis_tokens=vision.float(), z0=noise, nfe=1
        )
        return prepare_stack_actions(actions, self.execute_steps), {}


def get_model(cfg: DictConfig, torch_dtype=torch.float32) -> DEMPolicy:
    """Load an upstream dem-v1 file plus the required deployment manifest."""
    from dem import DEM

    if torch_dtype != torch.float32:
        raise ValueError("DEM head must remain FP32; set actor/rollout precision=fp32")
    path = Path(cfg.model_path)
    with path.with_suffix(".json").open() as stream:
        manifest = json.load(stream)
    if not manifest.get("contract_verified", False) and not cfg.dem.get(
        "allow_unverified_contract", False
    ):
        raise ValueError(
            "Data/control contract is unverified; explicit dem.allow_unverified_contract=true is debug-only"
        )
    encoding = manifest.get("state_rotation_encoding")
    if encoding not in STATE_ROTATION_ENCODINGS:
        raise ValueError(
            "DEM manifest needs an explicit supported state_rotation_encoding"
        )
    required = {
        "state_rotation": "euler_xyz",
        "gripper_state": "left_negative_right",
        "action_space": "isaaclab_raw_relative_ik_scale_0.5",
        "camera_keys": ["main_images", "wrist_images"],
        "fps": 20,
        "nfe": 1,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"DEM checkpoint protocol mismatch: {key}")
    with path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != manifest.get("sha256"):
        raise ValueError("DEM policy SHA256 does not match deployment manifest")
    policy = DEM.load(path).float().eval()
    if (
        policy.config.proprio_dim,
        policy.config.action_dim,
        policy.config.num_cameras,
    ) != (8, 7, 2):
        raise ValueError("Checkpoint is not a two-camera Stack-Cube DEM policy")
    return DEMPolicy(policy, int(cfg.num_action_chunks), int(cfg.dem.seed), encoding)
