#!/usr/bin/env python

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import imageio
import torch


SCRIPT_PATH = Path(__file__).resolve()
RLINF_REPO_ROOT = SCRIPT_PATH.parents[2]
if str(RLINF_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_ROOT))

DEFAULT_ENV_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"
DEFAULT_LIGHT_FILE = (
    RLINF_REPO_ROOT / "rlinf/assets_isaaclab/all_setting/light_setting_50.jsonl"
)
DEFAULT_FIXED_CUBE_POSES = {
    "cube_1": ([0.4, 0.0, 0.0203], [0.0, 0.0, 0.0]),
    "cube_2": ([0.52, 0.08, 0.0203], [0.0, 0.0, 0.0]),
    "cube_3": ([0.6, -0.1, 0.0203], [0.0, 0.0, 0.0]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render stack-cube previews under DomeLight settings from a jsonl file."
    )
    parser.add_argument(
        "--light-file",
        type=Path,
        default=DEFAULT_LIGHT_FILE,
        help="Path to light-setting jsonl file.",
    )
    parser.add_argument(
        "--light-ids",
        nargs="*",
        default=None,
        help="Optional subset of light setting ids. Defaults to all records in file order.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory under which the output folder is created. Defaults to logs/<run-name>/<YYYYMMDD-HHMMSS>.",
    )
    parser.add_argument(
        "--run-name",
        default="stack_cube_light_scenarios",
        help="Readable suffix used for the default log directory name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. Overrides --log-dir and default log discovery.",
    )
    parser.add_argument(
        "--output-name",
        default="light_setting_50_preview",
        help="Output subdirectory name when --output-dir is not given.",
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--render-frames", type=int, default=6)
    parser.add_argument("--save-wrist", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_light_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise KeyError(f"Missing light id at {path}:{line_no}")
            if "light" not in record:
                raise KeyError(f"Missing light payload at {path}:{line_no}")
            records.append(record)
    if not records:
        raise ValueError(f"No light records found in {path}")
    return records


def select_light_records(records, light_ids):
    if not light_ids:
        return records
    wanted = {str(light_id) for light_id in light_ids}
    selected = [record for record in records if str(record["id"]) in wanted]
    found = {str(record["id"]) for record in selected}
    missing = sorted(wanted - found)
    if missing:
        raise KeyError(f"Light ids not found: {missing}")
    return selected


def current_run_log_dir(log_root: Path, run_name: str):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    clean_run_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(run_name)
    ).strip("_")
    if clean_run_name:
        return log_root / clean_run_name / timestamp
    return log_root / timestamp


def resolve_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir
    if args.log_dir is not None:
        return args.log_dir / args.output_name
    return current_run_log_dir(RLINF_REPO_ROOT / "logs", args.run_name) / args.output_name


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    return _log


def texture_registry():
    assets_root = (
        RLINF_REPO_ROOT.parent
        / "IsaacAssets5.1/Assets/Isaac/5.1"
    )
    nvidia_root = assets_root / "NVIDIA"
    isaac_root = assets_root / "Isaac"

    return {
        "default": "",
        "abandoned_parking": str(nvidia_root / "Assets/AnimGraph/Worlds/textures/WarehouseInterior2b_4x8k.hdr"),
        "evening_road": str(nvidia_root / "Assets/AnimGraph/105.0/Worlds/textures/adams_place_bridge_4k.hdr"),
        "lakeside": str(isaac_root / "Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"),
        "autoshop": str(nvidia_root / "Assets/AnimGraph/Worlds/textures/ZetoCG_WarehouseInterior2b.hdr"),
        "carpentry_shop": str(nvidia_root / "Assets/AnimGraph/Characters/Reallusion/Debra/Props/Stage/B2_2k_256_4k.hdr"),
        "hospital_room": str(nvidia_root / "Assets/ArchVis/Industrial/Stages/HDRI/CloudyDay_04_04.hdr"),
        "hotel_room": str(isaac_root / "Environments/Outdoor/Rivermark/dsready_content/nv_core/common_tools/content_tagging/studio_lights/Materials/photo_studio_01_4k.hdr"),
        "small_empty_house": str(isaac_root / "Environments/Outdoor/Rivermark/dsready_content/nv_core/common_assets/environments/cloudy_sky/SubUSDs/textures/stars_4k.hdr"),
        "photo_studio": str(isaac_root / "Environments/Outdoor/Rivermark/dsready_content/nv_core/common_tools/content_tagging/studio_lights/Materials/photo_studio_01_4k.hdr"),
    }


def resolve_texture(light_cfg):
    explicit_texture = str(light_cfg.get("texture") or "")
    if explicit_texture:
        return explicit_texture

    key = str(light_cfg.get("texture_key", "default"))
    registry = texture_registry()
    if key not in registry:
        raise KeyError(f"Unknown texture_key={key}. Available: {sorted(registry)}")
    texture = registry[key]
    if texture and (not Path(texture).is_file() or Path(texture).stat().st_size <= 0):
        raise FileNotFoundError(
            f"DomeLight texture for texture_key={key} does not exist or is empty: {texture}"
        )
    return texture


def attr_value(prim, attr_name):
    attr = prim.GetAttribute(attr_name)
    return attr.Get() if attr else None


def jsonable_attr_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return [float(item) for item in value]
    except TypeError:
        return str(value)


def apply_light(env, record):
    light_cfg = dict(record.get("light", {}))
    if "intensity" not in light_cfg:
        raise KeyError(f"Missing light.intensity for id={record['id']}")
    if "color" not in light_cfg:
        raise KeyError(f"Missing light.color for id={record['id']}")

    intensity = float(light_cfg["intensity"])
    color = tuple(float(value) for value in light_cfg["color"])
    if len(color) != 3:
        raise ValueError(f"light.color must have 3 values for id={record['id']}")
    texture = resolve_texture(light_cfg)

    light = env.scene["light"]
    light_prim = light.prims[0]
    light_prim.GetAttribute("inputs:intensity").Set(intensity)
    light_prim.GetAttribute("inputs:color").Set(color)
    light_prim.GetAttribute("inputs:texture:file").Set(texture)

    return {
        "id": str(record["id"]),
        "preset": light_cfg.get("preset"),
        "requested_intensity": intensity,
        "requested_color": list(color),
        "requested_texture_key": light_cfg.get("texture_key", "default"),
        "requested_texture": light_cfg.get("texture", ""),
        "resolved_texture": texture,
        "light_prim_path": str(light_prim.GetPath()),
        "actual_intensity": jsonable_attr_value(attr_value(light_prim, "inputs:intensity")),
        "actual_color": jsonable_attr_value(attr_value(light_prim, "inputs:color")),
        "actual_texture": jsonable_attr_value(attr_value(light_prim, "inputs:texture:file")),
    }


def apply_cube_pose(env, env_id: int, asset_name: str, pos_xyz, rpy):
    import isaaclab.utils.math as math_utils

    asset = env.scene[asset_name]
    pose_tensor = torch.tensor([[*pos_xyz, *rpy]], device=env.device)
    positions = pose_tensor[:, 0:3] + env.scene.env_origins[env_id, 0:3]
    orientations = math_utils.quat_from_euler_xyz(
        pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
    )
    env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)


def apply_fixed_cube_poses(env):
    for asset_name, (pos_xyz, rpy) in DEFAULT_FIXED_CUBE_POSES.items():
        apply_cube_pose(env, 0, asset_name, pos_xyz, rpy)


def image_from_obs(obs, key: str):
    image = obs["policy"][key][0].detach().cpu().numpy()
    return image.astype("uint8")


def configure_env_cfg(env_cfg, args):
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

    env_cfg.seed = 0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.wrist_cam.height = args.height
    env_cfg.scene.wrist_cam.width = args.width
    env_cfg.scene.table_cam.height = args.height
    env_cfg.scene.table_cam.width = args.width
    if hasattr(env_cfg.events, "randomize_table_visual_material"):
        env_cfg.events.randomize_table_visual_material.func = franka_stack_events.noop_event
        env_cfg.events.randomize_table_visual_material.params = {}
    if hasattr(env_cfg.events, "randomize_robot_arm_visual_texture"):
        env_cfg.events.randomize_robot_arm_visual_texture.func = franka_stack_events.noop_event
        env_cfg.events.randomize_robot_arm_visual_texture.params = {}


def render_record(env, args, record, output_dir: Path, log):
    light_id = str(record["id"])
    log(f"render light_id={light_id}")
    env.reset()
    apply_fixed_cube_poses(env)
    applied_light = apply_light(env, record)

    for _ in range(max(int(args.render_frames), 1)):
        env.sim.render()
    table_cam = env.scene["table_cam"]
    if hasattr(table_cam, "reset"):
        table_cam.reset([0])
    obs = env.observation_manager.compute(update_history=True)

    table_png = output_dir / f"light_{light_id}_table.png"
    imageio.imwrite(table_png, image_from_obs(obs, "table_cam"))
    log(f"saved {table_png}")

    wrist_png = None
    if args.save_wrist:
        wrist_png = output_dir / f"light_{light_id}_wrist.png"
        imageio.imwrite(wrist_png, image_from_obs(obs, "wrist_cam"))
        log(f"saved {wrist_png}")

    payload = {
        "light_id": light_id,
        "record": record,
        "fixed_cube_poses": DEFAULT_FIXED_CUBE_POSES,
        "applied_light": applied_light,
        "table_png": str(table_png),
        "wrist_png": str(wrist_png) if wrist_png is not None else None,
    }
    debug_path = output_dir / f"light_{light_id}.debug.json"
    debug_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.debug:
        log("debug " + json.dumps(payload, ensure_ascii=False))
    return payload


def main():
    args = parse_args()
    records = select_light_records(load_light_records(args.light_file), args.light_ids)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(output_dir / "render_light.log")

    log(f"light_file={args.light_file}")
    log(f"output_dir={output_dir}")
    log(f"light_count={len(records)}")
    log(
        "args="
        + json.dumps(
            {
                "env_id": args.env_id,
                "width": args.width,
                "height": args.height,
                "render_frames": args.render_frames,
                "save_wrist": args.save_wrist,
                "run_name": args.run_name,
            },
            ensure_ascii=False,
        )
    )

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    env = None
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(args.env_id, "env_cfg_entry_point")
        configure_env_cfg(env_cfg, args)

        log(f"creating env env_id={args.env_id}")
        env = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
        manifest_path = output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest_fp:
            for idx, record in enumerate(records, 1):
                try:
                    log(f"render {idx}/{len(records)} light_id={record['id']}")
                    payload = render_record(env, args, record, output_dir, log)
                    manifest_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    manifest_fp.flush()
                except Exception:
                    log(f"failed light_id={record['id']}")
                    log(traceback.format_exc())
                    raise
        log(f"manifest={manifest_path}")
    finally:
        if env is not None:
            log("closing env")
            env.close()
        log("closing app")
        app.close()

    print(f"Saved light previews to: {output_dir}")


if __name__ == "__main__":
    main()
