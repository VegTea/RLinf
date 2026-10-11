# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""GPU regression: original pi05 train config, startup/reset images and counters.

Use one approved GPU. This starts only two environments, no policy or RL workers.
A fresh process is required per invocation. Warmup=0 reproduces the old failure.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from scipy.spatial.transform import Rotation

from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv


def color_centroids(image: np.ndarray) -> np.ndarray:
    """Locate red/green/blue cube pixel centroids on the unannotated table image."""
    image = image.astype(float)
    centers = []
    for channel in range(3):
        mask = (image[..., channel] > 40) & (
            image[..., channel]
            > 1.8 * np.max(np.delete(image, channel, axis=2), axis=2)
        )
        if mask.sum() < 100:
            raise AssertionError("Cube color missing/occluded in reset image")
        centers.append(np.argwhere(mask).mean(axis=0)[::-1])
    return np.asarray(centers)


class ResetProbe(IsaaclabStackCubeEnv):
    """Audit native reset results before they enter the RLinf observation wrapper."""

    def _make_env_function(self):
        original = super()._make_env_function()
        out = Path(self.cfg.reset_probe_output)

        def factory():
            env, app = original()
            reset = env.reset
            counter = 0

            def array(value):
                if hasattr(value, "torch"):
                    value = value.torch
                return value.detach().cpu().numpy().copy()

            def wrapped(*args, **kwargs):
                nonlocal counter
                step_before = env.sim._physics_step_count
                q_before = array(env.scene["robot"].data.joint_pos)
                obs, info = reset(*args, **kwargs)
                assert env.sim._physics_step_count == step_before
                ids = kwargs.get("env_ids")
                ids = list(range(env.num_envs)) if ids is None else ids.cpu().tolist()
                assert not env.episode_length_buf[ids].any()
                assert not array(env.scene["robot"].data.joint_vel)[ids].any()
                remaining = [i for i in range(env.num_envs) if i not in ids]
                np.testing.assert_array_equal(
                    array(env.scene["robot"].data.joint_pos)[remaining],
                    q_before[remaining],
                )
                camera = env.scene["table_cam"].data
                positions = array(camera.pos_w)
                quaternions = array(camera.quat_w_ros)
                intrinsics = array(camera.intrinsic_matrices)
                # IsaacLab's standard assets: cube_2 red, cube_3 green, cube_1 blue.
                cubes = np.stack(
                    [
                        array(env.scene[name].data.root_pos_w)
                        for name in ("cube_2", "cube_3", "cube_1")
                    ],
                    axis=1,
                )
                errors = []
                for i in range(env.num_envs):
                    image = obs["policy"]["table_cam"][i].cpu().numpy().copy()
                    imageio.imwrite(out / f"reset_{counter:02d}_env_{i}.png", image)
                    camera_points = (
                        Rotation.from_quat(quaternions[i])
                        .inv()
                        .apply(cubes[i] - positions[i])
                    )
                    assert (camera_points[:, 2] > 0).all()
                    pixels = camera_points @ intrinsics[i].T
                    pixels = pixels[:, :2] / pixels[:, 2:]
                    errors.append(
                        np.linalg.norm(color_centroids(image) - pixels, axis=1).tolist()
                    )
                report = {
                    "seed": kwargs.get("seed"),
                    "reset_ids": ids,
                    "physics_step_before_and_after": step_before,
                    "episode_steps": env.episode_length_buf.cpu().tolist(),
                    "cube_projection_errors_px": errors,
                    "cube_positions": cubes.tolist(),
                    "joint_positions": array(
                        env.scene["robot"].data.joint_pos
                    ).tolist(),
                    "unreset_joints_unchanged": True,
                }
                (out / f"reset_{counter:02d}.json").write_text(
                    json.dumps(report, indent=2)
                )
                counter += 1
                print("RESET_AUDIT", json.dumps(report), flush=True)
                return obs, info

            env.reset = wrapped
            return env, app

        return factory


def main() -> None:
    """Check consecutive full resets, then a partial reset after two control steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-steps", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.multiprocessing.set_start_method("spawn", force=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated/visible GPU is required")
    root = Path(__file__).resolve().parents[2]
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    with initialize_config_dir(config_dir=str(root / "examples/embodiment/config")):
        full = compose(
            config_name="isaaclab_franka_stack_cube_ppo_openpi_pi05_isaacsim6_chunkopt"
        )
    cfg = full.env.train
    with open_dict(cfg):
        cfg.reset_probe_output = str(out)
        cfg.init_params.startup_physics_warmup_steps = args.warmup_steps
    env = ResetProbe(cfg, 2, 0, 1, SimpleNamespace(rank=0))
    try:
        for seed in range(42, 47):
            env.reset(seed=seed)
            assert not env.elapsed_steps.any()
        for _ in range(2):
            env.step(torch.zeros(2, 7), auto_reset=False)
        env.reset(seed=47, env_ids=torch.tensor([0], device=env.device))
    finally:
        env.close()
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("reset_*.json"))]
    maximum = max(np.max(row["cube_projection_errors_px"]) for row in rows)
    summary = {
        "warmup_steps": args.warmup_steps,
        "reset_calls": len(rows),
        "max_cube_projection_error_px": float(maximum),
        "first_reset_physics_step": rows[0]["physics_step_before_and_after"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.warmup_steps:
        assert maximum < 5, summary
        assert all(row["physics_step_before_and_after"] == 1 for row in rows[:-1])
        assert rows[-1]["episode_steps"] == [0, 2]
    else:
        assert np.max(rows[0]["cube_projection_errors_px"]) > 20, summary
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
