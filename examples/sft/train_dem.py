# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Small real-data DEM SFT, independent of RLinf's PPO/FSDP actor workers."""

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter

from rlinf.data.datasets.dem_stack_cube import StackCubeDataset
from rlinf.models.embodiment.dem.stack_cube_io import (
    DATASET_REVISION,
    DEM_REVISION,
    STATE_ROTATION_ENCODINGS,
)

logger = logging.getLogger(__name__)


def save_policy(policy, path: Path, metadata: dict) -> None:
    """Atomically publish dem-v1 weights, then a checksum-bearing manifest."""
    temporary = path.with_suffix(".pt.tmp")
    policy.save(temporary, include_frozen=True)
    temporary.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {**metadata, "sha256": digest}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path.with_suffix(".json"))


def main() -> None:
    """Train the upstream objective on sampled, unpadded demonstration chunks."""
    from dem import DEM, DEMConfig, Normalizer, preprocess_images

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", default="0,1,2,3")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vision-model", required=True, help="Authorized local DINOv3 directory"
    )
    parser.add_argument(
        "--language-model", required=True, help="Local NeoBERT directory"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="New run directory under pretrained_models/DEM/StackCube",
    )
    parser.add_argument("--resume", help="Trusted training.pt from this script")
    parser.add_argument(
        "--allow-unverified-contract",
        action="store_true",
        help="Debug only: do not claim control semantics verified",
    )
    parser.add_argument(
        "--state-rotation-encoding",
        required=True,
        choices=STATE_ROTATION_ENCODINGS,
        help="Explicit native-quaternion to stored-Euler hypothesis; does not transform dataset rows",
    )
    args = parser.parse_args()
    if not args.allow_unverified_contract:
        parser.error(
            "Acquisition provenance is not yet verified; debug training requires --allow-unverified-contract"
        )
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Run real DEM training on an approved GPU, not the CPU preparation host"
        )
    for model_path in (args.vision_model, args.language_model):
        if not Path(model_path, "model.safetensors").is_file():
            raise FileNotFoundError(f"Pretrained model weights missing: {model_path}")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite an existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    generator = np.random.default_rng(args.seed)
    episodes = [int(value) for value in args.episodes.split(",")]
    dataset = StackCubeDataset(args.data, episodes, horizon=16)
    stats = dataset.statistics()
    cfg = DEMConfig(
        action_dim=7,
        proprio_dim=8,
        chunk=16,
        num_cameras=2,
        vision_model_name=args.vision_model,
        language_model=args.language_model,
        freeze_vision=False,
        freeze_language=True,
    )
    policy = DEM(cfg).cuda()
    policy.action_norm = Normalizer.from_mean_std(
        stats["action_mean"], stats["action_std"]
    ).cuda()
    policy.proprio_norm = Normalizer.from_mean_std(
        stats["state_mean"], stats["state_std"]
    ).cuda()
    optimizer = torch.optim.AdamW(
        policy.parameter_groups(1e-4, 5e-5), weight_decay=1e-4, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    start = 0
    if args.resume:
        # Resume files contain Python RNG objects; accept only the user's trusted run.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["episodes"] != episodes or saved["total_steps"] != args.steps:
            raise ValueError(
                "Resume requires identical episode selection and total steps"
            )
        if saved.get("state_rotation_encoding") != args.state_rotation_encoding:
            raise ValueError("Resume state rotation encoding differs or is missing")
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.bit_generator.state = saved["numpy_generator"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["step"]
    metadata = {
        "state_rotation": "euler_xyz",
        "state_rotation_encoding": args.state_rotation_encoding,
        "gripper_state": "left_negative_right",
        "action_space": "isaaclab_raw_relative_ik_scale_0.5",
        "camera_keys": ["main_images", "wrist_images"],
        "fps": 20,
        "nfe": 1,
        "prediction_horizon": 16,
        "default_execute_steps": 5,
        "dataset_revision": DATASET_REVISION,
        "dem_revision": DEM_REVISION,
        "episodes": episodes,
        "seed": args.seed,
        "contract_verified": False,
        "pretrained_vision": True,
        "pretrained_language": True,
        "training_args": vars(args),
    }
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    writer = SummaryWriter(str(out / "tensorboard"), purge_step=start or None)
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    policy.train()
    with (out / "metrics.jsonl").open("a") as log:
        for step in range(start, args.steps):
            batch = default_collate(
                [
                    dataset[int(i)]
                    for i in generator.integers(len(dataset), size=args.batch_size)
                ]
            )
            images = preprocess_images(batch["images"], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vision = policy.encode_vision(images)
            loss, aux = policy.loss(
                None,
                batch["proprio"],
                batch["actions"],
                texts=batch["instruction"],
                vis_tokens=vision.float(),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            metrics = {
                "step": step + 1,
                "loss": loss.item(),
                "grad_norm": float(norm),
                **aux,
            }
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            for name, value in metrics.items():
                if name != "step":
                    writer.add_scalar(f"train/{name}", value, step + 1)
            if (step + 1) % 10 == 0 or step == 0:
                logger.info("%s", metrics)
            if (step + 1) % 100 == 0 or step + 1 == args.steps:
                save_policy(
                    policy,
                    out / f"policy_step_{step + 1}.pt",
                    {**metadata, "step": step + 1},
                )
                temporary = out / "training.pt.tmp"
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step + 1,
                        "total_steps": args.steps,
                        "episodes": episodes,
                        "state_rotation_encoding": args.state_rotation_encoding,
                        "numpy_generator": generator.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    },
                    temporary,
                )
                temporary.replace(out / "training.pt")
    writer.close()


if __name__ == "__main__":
    main()
