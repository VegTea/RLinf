# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""GPU-only single-env protocol probe; not a trained-policy evaluation."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv
from rlinf.models.embodiment.dem.stack_cube_io import environment_state_to_dataset


class ProbeEnv(IsaaclabStackCubeEnv):
    """Expose existing raw observations only within this diagnostic process."""

    def _wrap_obs(self, obs):
        result = super()._wrap_obs(obs)
        result["raw_quat_xyzw"] = result["eef_quat_xyzw"]
        result["raw_previous_action"] = obs["policy"]["actions"].clone()
        return result


def snapshot(obs: dict) -> dict:
    """Compare legacy wrapper output with IsaacLab 3's native XYZW orientation."""
    legacy = obs["states"].detach().float().cpu().numpy()[0].copy()
    legacy[3:6] = Rotation.from_rotvec(legacy[3:6]).as_euler("xyz")
    quat = obs["raw_quat_xyzw"].cpu().numpy()[0]
    native = Rotation.from_quat(quat)
    state = (
        environment_state_to_dataset(
            obs["states"], obs["eef_quat_xyzw"], encoding="native_xyzw"
        )
        .cpu()
        .numpy()[0]
    )
    error = float((Rotation.from_euler("xyz", state[3:6]).inv() * native).magnitude())
    if error > 1e-6:
        raise AssertionError(f"Native rotation round trip failed: {error}")
    candidate = (
        environment_state_to_dataset(
            obs["states"], obs["eef_quat_xyzw"], encoding="legacy_wxyz_as_xyzw"
        )
        .cpu()
        .numpy()[0]
    )
    return {
        "state_native_euler_xyz": state.tolist(),
        "legacy_adapter_state": legacy.tolist(),
        "candidate_wxyz_as_xyzw_state": candidate.tolist(),
        "raw_quat_xyzw": quat.tolist(),
        "legacy_vs_native_rotation_error_rad": float(
            (Rotation.from_euler("xyz", legacy[3:6]).inv() * native).magnitude()
        ),
        "rotation_roundtrip_error_rad": error,
        "previous_action": obs["raw_previous_action"].cpu().tolist()[0],
    }


def main() -> None:
    """Trace all six pose axes, gripper signs, and optional raw-demo prefix replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--replay",
        action="append",
        help="NPZ with real states/actions; repeat for multiple episodes",
    )
    parser.add_argument("--replay-steps", type=int, default=64)
    parser.add_argument(
        "--replay-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    args = parser.parse_args()
    if args.replay_steps < 1:
        parser.error("--replay-steps must be positive")
    if not all(np.isfinite(s) and 0 < s <= 2 for s in args.replay_scales):
        parser.error("--replay-scales must be finite and in (0, 2]")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.multiprocessing.set_start_method("spawn", force=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated/visible GPU is required")
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root / "examples/embodiment/config/env/isaaclab_stack_cube.yaml"
    )
    cfg.init_params.max_episode_steps = 450
    cfg.max_episode_steps = 450
    cfg.video_cfg.video_base_dir = str(out)
    cfg.init_params.enable_depth_observations = False
    cfg.init_params.antialiasing_mode = "DLAA"
    cfg.init_params.ik_diagnostic_path = str(out / "ik_trace.jsonl")
    cfg.seed = 42
    cfg.include_native_eef_quat = True
    env = ProbeEnv(cfg, 1, 0, 1, SimpleNamespace(rank=0))
    report = {"purpose": "control contract probe, not DEM evaluation", "trials": []}
    try:
        obs, _ = env.reset(seed=42)
        report["reset"] = snapshot(obs)
        report["cameras"] = {}
        for key in ("main_images", "wrist_images"):
            image = obs[key][0].cpu().numpy()
            imageio.imwrite(out / f"{key}.png", image)
            report["cameras"][key] = {
                "shape": list(image.shape),
                "dtype": str(image.dtype),
            }
        trace_step = 0

        def run_trial(name: str, actions: np.ndarray) -> None:
            nonlocal trace_step
            obs, _ = env.reset(seed=42)
            trial = {"name": name, "initial": snapshot(obs), "steps": []}
            report["trials"].append(trial)
            for action in actions:
                before = snapshot(obs)
                obs, reward, terminated, truncated, _ = env.step(
                    torch.tensor(action[None], dtype=torch.float32), auto_reset=False
                )
                trace_step += 1
                after = snapshot(obs)
                # Observation.actions must describe the action just applied.
                np.testing.assert_allclose(after["previous_action"], action, atol=1e-6)
                trial["steps"].append(
                    {
                        "trace_step": trace_step,
                        "raw_action": action.tolist(),
                        "before": before,
                        "after": after,
                        "reward": reward.cpu().tolist(),
                        "terminated": terminated.cpu().tolist(),
                        "truncated": truncated.cpu().tolist(),
                    }
                )
                if bool(terminated.any() or truncated.any()):
                    raise RuntimeError(f"Unexpected inner episode boundary in {name}")
            (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"PROBE completed {name}: {len(actions)} steps", flush=True)

        for axis in range(6):
            for sign in (-1, 1):
                actions = np.zeros((8, 7), dtype=np.float32)
                actions[:, -1] = 1
                actions[3:6, axis] = sign * 0.02
                run_trial(f"axis_{axis}_sign_{sign}", actions)
        actions = np.zeros((45, 7), dtype=np.float32)
        actions[:15, -1] = 1
        actions[15:30, -1] = -1
        # The final zero command is intentionally tested as open, not hold.
        run_trial("gripper_open_close_zero", actions)
        for replay in args.replay or []:
            with np.load(replay, allow_pickle=False) as data:
                actions = data["actions"][: args.replay_steps].copy()
                states = data["states"][: len(actions) + 1].copy()
            if (
                actions.ndim != 2
                or actions.shape[1] != 7
                or not np.isfinite(actions).all()
            ):
                raise ValueError("Expected finite demo actions [T,7]")
            for scale in args.replay_scales:
                scaled = actions.copy()
                scaled[:, :6] *= scale
                run_trial(f"demo_{Path(replay).stem}_multiplier_{scale}", scaled)
                report["trials"][-1]["demo_states"] = states.tolist()
                report["trials"][-1]["demo_source"] = str(Path(replay).resolve())
        report["complete"] = True
        report["contract_verified"] = False
        report["limitation"] = (
            "Runtime checks do not establish missing acquisition/conversion provenance."
        )
        (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
