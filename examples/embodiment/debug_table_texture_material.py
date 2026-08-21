import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import imageio
import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
RLINF_REPO_ROOT = SCRIPT_PATH.parents[2]
if str(RLINF_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_ROOT))


TEXTURE_ATTRS = [
    "inputs:diffuse_texture",
    "inputs:normalmap_texture",
    "inputs:normal_texture",
    "inputs:metallic_texture",
    "inputs:reflectionroughness_texture",
    "inputs:roughness_texture",
    "inputs:orm_texture",
    "inputs:occlusionroughnessmetallic_texture",
    "inputs:occlusion_texture",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug the legacy runtime table texture replacement path without swapping table USD assets."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for PNG/JSON/log outputs.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0",
    )
    parser.add_argument(
        "--texture-keys",
        type=str,
        nargs="+",
        default=["copper", "brass", "steel_stainless"],
        help="Keys from build_table_texture_registry() to apply in order.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--render-frames", type=int, default=8)
    parser.add_argument("--debug-texture-application", action="store_true")
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

    table_cam = env.scene["table_cam"]
    table_cam.reset([0])
    obs = env.observation_manager.compute(update_history=True)
    return obs["policy"]["table_cam"][0].detach().cpu().numpy().astype("uint8")


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def walk_prim_tree(prim):
    yield prim
    for child in prim.GetChildren():
        yield from walk_prim_tree(child)


def collect_table_texture_debug(env):
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    table_asset = env.scene["table"]
    debug = {
        "table_prim_paths": [str(path) for path in table_asset.prim_paths],
        "shaders": [],
        "prototype_shaders": [],
    }

    for env_id, table_root in enumerate(table_asset.prim_paths):
        root_prim = stage.GetPrimAtPath(str(table_root))
        if not root_prim or not root_prim.IsValid():
            debug["shaders"].append(
                {"env_id": env_id, "table_root": str(table_root), "is_valid": False}
            )
            continue

        for prim in walk_prim_tree(root_prim):
            if prim.GetTypeName() != "Shader":
                continue
            shader_info = {
                "env_id": env_id,
                "path": prim.GetPath().pathString,
                "is_instance_proxy": bool(prim.IsInstanceProxy()),
                "textures": {},
            }
            for attr_name in TEXTURE_ATTRS:
                attr = prim.GetAttribute(attr_name)
                if attr:
                    shader_info["textures"][attr_name] = str(attr.Get())
            debug["shaders"].append(shader_info)

        visuals_prim = stage.GetPrimAtPath(f"{table_root}/Visuals")
        if not visuals_prim or not visuals_prim.IsValid() or not visuals_prim.IsInstance():
            continue
        prototype = visuals_prim.GetPrototype()
        if not prototype or not prototype.IsValid():
            continue
        for prim in walk_prim_tree(prototype):
            if prim.GetTypeName() != "Shader":
                continue
            shader_info = {
                "env_id": env_id,
                "visuals_path": f"{table_root}/Visuals",
                "prototype_path": prototype.GetPath().pathString,
                "path": prim.GetPath().pathString,
                "textures": {},
            }
            for attr_name in TEXTURE_ATTRS:
                attr = prim.GetAttribute(attr_name)
                if attr:
                    shader_info["textures"][attr_name] = str(attr.Get())
            debug["prototype_shaders"].append(shader_info)
    return debug


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(output_dir / "debug_table_texture_material.log")
    report = {
        "script": "debug_table_texture_material.py",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_id": args.env_id,
        "texture_keys": list(args.texture_keys),
        "render_frames": int(args.render_frames),
        "results": [],
    }

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    env = None
    try:
        import gymnasium as gym
        from isaaclab.managers import SceneEntityCfg
        from isaaclab_tasks.manager_based.manipulation.stack.mdp import (
            franka_stack_events,
        )
        from isaaclab_tasks.utils import load_cfg_from_registry
        from rlinf.envs.isaaclab.scenario_loader import build_table_texture_registry

        texture_registry = build_table_texture_registry()
        report["available_texture_keys"] = sorted(texture_registry)
        missing_keys = [key for key in args.texture_keys if key not in texture_registry]
        if missing_keys:
            raise KeyError(
                f"Missing texture keys: {missing_keys}. Available: {sorted(texture_registry)}"
            )

        env_cfg = load_cfg_from_registry(args.env_id, "env_cfg_entry_point")
        env_cfg.seed = 0
        env_cfg.scene.num_envs = 1
        env_cfg.scene.wrist_cam.height = args.height
        env_cfg.scene.wrist_cam.width = args.width
        env_cfg.scene.table_cam.height = args.height
        env_cfg.scene.table_cam.width = args.width
        if hasattr(env_cfg.events, "randomize_table_visual_material"):
            env_cfg.events.randomize_table_visual_material.func = (
                franka_stack_events.noop_event
            )
            env_cfg.events.randomize_table_visual_material.params = {}
        if hasattr(env_cfg.events, "randomize_robot_arm_visual_texture"):
            env_cfg.events.randomize_robot_arm_visual_texture.func = (
                franka_stack_events.noop_event
            )
            env_cfg.events.randomize_robot_arm_visual_texture.params = {}

        log("creating gym env for table texture debug")
        env = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
        env._debug_texture_application = bool(args.debug_texture_application)
        env.reset()

        baseline_image = render_table_image(env, args.render_frames)
        baseline_path = output_dir / "baseline.png"
        save_image(baseline_path, baseline_image)
        report["baseline"] = {
            "image_path": str(baseline_path),
            "texture_debug": collect_table_texture_debug(env),
        }

        previous_image = baseline_image
        for texture_key in args.texture_keys:
            texture_path = texture_registry[texture_key]
            log(f"applying table texture: key={texture_key} path={texture_path}")
            before = collect_table_texture_debug(env)
            franka_stack_events.apply_visual_texture_material(
                env=env,
                env_ids=torch.tensor([0], device=env.device),
                asset_cfg=SceneEntityCfg("table"),
                texture=texture_path,
            )
            after = collect_table_texture_debug(env)
            image = render_table_image(env, args.render_frames)
            image_path = output_dir / f"texture_{texture_key}.png"
            save_image(image_path, image)
            mean_abs_diff = float(
                np.mean(np.abs(image.astype(np.float32) - previous_image.astype(np.float32)))
            )
            report["results"].append(
                {
                    "texture_key": texture_key,
                    "texture_path": texture_path,
                    "image_path": str(image_path),
                    "mean_abs_image_diff_vs_previous": mean_abs_diff,
                    "before": before,
                    "after": after,
                }
            )
            previous_image = image

        report["status"] = "ok"
    except Exception:
        report["status"] = "error"
        report["traceback"] = traceback.format_exc()
        log("fatal exception during table texture debug")
        log(report["traceback"])
        raise
    finally:
        report_path = output_dir / "debug_table_texture_material.json"
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
