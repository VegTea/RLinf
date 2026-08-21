import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import imageio
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
RLINF_REPO_ROOT = SCRIPT_PATH.parents[2]
if str(RLINF_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimental debug script for swapping SeattleLabTable USD references at reset time."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for debug PNG/JSON/log outputs.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0",
        help="Gym environment id.",
    )
    parser.add_argument(
        "--table-assets",
        type=str,
        nargs="+",
        required=True,
        help="List of table asset filenames or absolute paths to alternate between.",
    )
    parser.add_argument(
        "--num-swaps",
        type=int,
        default=6,
        help="How many swap attempts to run after the initial reset.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--render-frames",
        type=int,
        default=4,
        help="How many render ticks to run after each reset/swap.",
    )
    parser.add_argument(
        "--step-after-swap",
        type=int,
        default=2,
        help="How many zero-action env steps to run after each swap.",
    )
    parser.add_argument(
        "--image-diff-threshold",
        type=float,
        default=2.0,
        help="Mean absolute pixel difference threshold for marking a visible image change.",
    )
    return parser.parse_args()


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    return _log


def render_table_image(env, render_frames: int):
    for _ in range(max(int(render_frames), 1)):
        env.sim.render()
    obs = env.observation_manager.compute(update_history=True)
    image = obs["policy"]["table_cam"][0].detach().cpu().numpy()
    return image.astype("uint8")


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def collect_table_prim_debug(stage, table_prim_path: str):
    table_prim = stage.GetPrimAtPath(table_prim_path)
    debug = {
        "table_prim_path": table_prim_path,
        "is_valid": bool(table_prim and table_prim.IsValid()),
        "children": [],
        "references": [],
    }
    if not table_prim or not table_prim.IsValid():
        return debug

    debug["type_name"] = table_prim.GetTypeName()
    debug["active"] = bool(table_prim.IsActive())
    debug["instanceable"] = bool(table_prim.IsInstanceable())
    debug["child_count"] = len(table_prim.GetChildren())
    for child in table_prim.GetChildren():
        debug["children"].append(
            {"path": child.GetPath().pathString, "type_name": child.GetTypeName()}
        )

    prim_stack = table_prim.GetPrimStack()
    if prim_stack:
        reference_list = prim_stack[0].referenceList
        for item in reference_list.GetAddedOrExplicitItems():
            debug["references"].append(
                {
                    "asset_path": item.assetPath,
                    "prim_path": str(item.primPath),
                }
            )
    return debug


def swap_table_reference(stage, table_prim_path: str, target_asset_path: str):
    table_prim = stage.GetPrimAtPath(table_prim_path)
    if not table_prim or not table_prim.IsValid():
        raise RuntimeError(f"Invalid table prim for swap: {table_prim_path}")

    references = table_prim.GetReferences()
    references.ClearReferences()
    references.AddReference(target_asset_path)
    return collect_table_prim_debug(stage, table_prim_path)


def zero_action_for_env(env):
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None
    total_action_dim = 0
    for term in action_manager._terms.values():
        total_action_dim += int(term.action_dim)
    import torch

    return torch.zeros((env.num_envs, total_action_dim), device=env.device)


def step_env_safely(env, step_count: int):
    if step_count <= 0:
        return []
    debug = []
    action = zero_action_for_env(env)
    if action is None:
        return debug

    for step_idx in range(step_count):
        obs, reward, terminated, truncated, info = env.step(action)
        debug.append(
            {
                "step_idx": step_idx,
                "reward_mean": float(reward.detach().float().mean().item()),
                "terminated_any": bool(terminated.detach().any().item()),
                "truncated_any": bool(truncated.detach().any().item()),
                "info_keys": sorted(info.keys()),
                "table_cam_shape": list(obs["policy"]["table_cam"].shape),
            }
        )
    return debug


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(output_dir / "debug_runtime_swap_table_asset.log")
    report = {
        "script": "debug_runtime_swap_table_asset.py",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_id": args.env_id,
        "num_swaps": int(args.num_swaps),
        "render_frames": int(args.render_frames),
        "step_after_swap": int(args.step_after_swap),
        "image_diff_threshold": float(args.image_diff_threshold),
        "swap_results": [],
    }

    from rlinf.envs.isaaclab.scenario_loader import resolve_table_asset_path

    resolved_assets = [
        resolve_table_asset_path(asset, must_exist=True) for asset in args.table_assets
    ]
    report["requested_assets"] = list(args.table_assets)
    report["resolved_assets"] = resolved_assets

    from isaaclab.app import AppLauncher

    log("creating AppLauncher for isolated runtime table swap debug")
    app = AppLauncher(headless=True, enable_cameras=True).app
    env = None

    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaacsim.core.utils.stage import get_current_stage

        env_cfg = load_cfg_from_registry(args.env_id, "env_cfg_entry_point")
        env_cfg.seed = 0
        env_cfg.scene.num_envs = 1
        env_cfg.scene.wrist_cam.height = args.height
        env_cfg.scene.wrist_cam.width = args.width
        env_cfg.scene.table_cam.height = args.height
        env_cfg.scene.table_cam.width = args.width
        env_cfg.scene.table.spawn.usd_path = resolved_assets[0]
        log(f"initial table asset for env creation: {resolved_assets[0]}")

        env = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
        log("created gym env, calling initial reset")
        env.reset()
        log("initial reset finished")

        stage = get_current_stage()
        table_asset = env.scene["table"]
        table_prim_path = str(table_asset.prim_paths[0])
        report["table_prim_path"] = table_prim_path

        baseline_image = render_table_image(env, args.render_frames)
        save_image(output_dir / "baseline.png", baseline_image)
        report["baseline"] = collect_table_prim_debug(stage, table_prim_path)
        prev_image = baseline_image

        for swap_idx in range(int(args.num_swaps)):
            target_asset_path = resolved_assets[(swap_idx + 1) % len(resolved_assets)]
            log(f"swap {swap_idx}: target_asset_path={target_asset_path}")
            result = {
                "swap_idx": swap_idx,
                "target_asset_path": target_asset_path,
            }
            try:
                env.reset()
                result["reset_before_swap"] = "ok"
                result["before_swap"] = collect_table_prim_debug(stage, table_prim_path)
                result["after_swap"] = swap_table_reference(
                    stage, table_prim_path, target_asset_path
                )
                image = render_table_image(env, args.render_frames)
                image_path = output_dir / f"swap_{swap_idx:03d}.png"
                save_image(image_path, image)
                result["image_path"] = str(image_path)
                diff_value = float(
                    np.mean(np.abs(image.astype(np.float32) - prev_image.astype(np.float32)))
                )
                result["mean_abs_image_diff_vs_previous"] = diff_value
                result["visible_change"] = diff_value >= float(args.image_diff_threshold)
                result["step_debug"] = step_env_safely(env, int(args.step_after_swap))
                result["status"] = "ok"
                prev_image = image
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
                result["traceback"] = traceback.format_exc()
                log(f"swap {swap_idx} failed: {exc}")
            report["swap_results"].append(result)

        report["successful_swaps"] = sum(
            1 for item in report["swap_results"] if item["status"] == "ok"
        )
        report["visible_change_swaps"] = sum(
            1
            for item in report["swap_results"]
            if item.get("visible_change") is True
        )
        report["all_swaps_ok"] = (
            report["successful_swaps"] == len(report["swap_results"])
        )
    except Exception:
        report["fatal_error"] = traceback.format_exc()
        log("fatal exception during runtime table swap debug")
        log(report["fatal_error"])
        raise
    finally:
        report_path = output_dir / "debug_runtime_swap_table_asset.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log(f"wrote report json: {report_path}")
        if env is not None:
            try:
                env.close()
            except Exception:
                log("env.close() failed during cleanup")
                log(traceback.format_exc())
        log("closing AppLauncher")
        app.close()


if __name__ == "__main__":
    main()
