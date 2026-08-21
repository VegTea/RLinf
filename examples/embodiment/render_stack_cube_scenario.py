import argparse
import json
import time
import traceback
from pathlib import Path

import imageio
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a single stack-cube scenario by ID without running the full eval pipeline."
    )
    parser.add_argument(
        "--scenario-file",
        type=str,
        required=True,
        help="Path to scenario jsonl file.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        required=True,
        help="Scenario ID to render.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0",
        help="Gym environment id.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed material/shader debug information and save a JSON report next to the output PNG.",
    )
    return parser.parse_args()


def load_scenario(path: Path, scenario_id: str):
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if str(record["id"]) == str(scenario_id):
                return record
    raise KeyError(f"Scenario id not found: {scenario_id}")


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    return _log


def apply_cube_pose(env, env_id: int, asset_name: str, pos_xyz, rpy):
    import isaaclab.utils.math as math_utils

    asset = env.scene[asset_name]
    pose_tensor = torch.tensor([[*pos_xyz, *rpy]], device=env.device)
    positions = pose_tensor[:, 0:3] + env.scene.env_origins[env_id, 0:3]
    orientations = math_utils.quat_from_euler_xyz(
        pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
    )
    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=torch.tensor([env_id], device=env.device),
    )
    asset.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=env.device),
        env_ids=torch.tensor([env_id], device=env.device),
    )


def collect_table_texture_debug(env):
    from isaacsim.core.utils.stage import get_current_stage

    table_asset = env.scene["table"]
    stage = get_current_stage()
    debug = {
        "table_prim_paths": list(table_asset.prim_paths),
        "shader_paths": [],
    }

    for env_id, table_root in enumerate(table_asset.prim_paths):
        shader_paths = [
            # 旧table usd调用
            # f"{table_root}/DemoTable/Visuals/Looks/table_base/Shader",
            # f"{table_root}/DemoTable/Visuals/Looks/table_parts/Shader",
            f"{table_root}/Visuals/Looks/table_base/Shader",
            f"{table_root}/Visuals/Looks/table_parts/Shader",
        ]
        for shader_path in shader_paths:
            shader_prim = stage.GetPrimAtPath(shader_path)
            shader_info = {
                "env_id": env_id,
                "shader_path": shader_path,
                "is_valid": bool(shader_prim and shader_prim.IsValid()),
            }
            if shader_prim and shader_prim.IsValid():
                attr = shader_prim.GetAttribute("inputs:diffuse_texture")
                shader_info["has_diffuse_attr"] = bool(attr)
                shader_info["diffuse_texture"] = str(attr.Get()) if attr else None
            debug["shader_paths"].append(shader_info)

    def walk_prim_tree(prim):
        yield prim
        for child in prim.GetChildren():
            yield from walk_prim_tree(child)

    debug["table_subtree"] = []
    debug["table_prototypes"] = []
    for env_id, table_root in enumerate(table_asset.prim_paths):
        root_prim = stage.GetPrimAtPath(table_root)
        if not root_prim or not root_prim.IsValid():
            debug["table_subtree"].append(
                {
                    "env_id": env_id,
                    "table_root": table_root,
                    "is_valid": False,
                }
            )
            continue

        for prim in walk_prim_tree(root_prim):
            prim_path = prim.GetPath().pathString
            entry = {
                "env_id": env_id,
                "path": prim_path,
                "type": prim.GetTypeName(),
                "is_instance": bool(prim.IsInstance()),
                "is_instance_proxy": bool(prim.IsInstanceProxy()),
                "is_prototype": bool(prim.IsPrototype()),
            }
            rel = prim.GetRelationship("material:binding")
            if rel and rel.GetTargets():
                entry["material_binding"] = [str(t) for t in rel.GetTargets()]
            if prim.GetTypeName() == "Shader":
                attr = prim.GetAttribute("inputs:diffuse_texture")
                if attr:
                    entry["diffuse_texture"] = str(attr.Get())
            debug["table_subtree"].append(entry)

        visuals_prim = stage.GetPrimAtPath(f"{table_root}/Visuals")
        if visuals_prim and visuals_prim.IsValid() and visuals_prim.IsInstance():
            prototype = visuals_prim.GetPrototype()
            proto_entry = {
                "env_id": env_id,
                "visuals_path": f"{table_root}/Visuals",
                "prototype_path": prototype.GetPath().pathString
                if prototype and prototype.IsValid()
                else None,
                "children": [],
            }
            if prototype and prototype.IsValid():
                for prim in walk_prim_tree(prototype):
                    item = {
                        "path": prim.GetPath().pathString,
                        "type": prim.GetTypeName(),
                    }
                    rel = prim.GetRelationship("material:binding")
                    if rel and rel.GetTargets():
                        item["material_binding"] = [str(t) for t in rel.GetTargets()]
                    if prim.GetTypeName() == "Shader":
                        attr = prim.GetAttribute("inputs:diffuse_texture")
                        if attr:
                            item["diffuse_texture"] = str(attr.Get())
                    proto_entry["children"].append(item)
            debug["table_prototypes"].append(proto_entry)
    return debug


def apply_table_texture(env, texture_key: str, debug: bool = False, log=None):
    # 旧运行时table材质替换逻辑。
    # 预览脚本已切换为在 env 创建前选择 table_asset，这个函数保留用于回退排查。
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
    from rlinf.envs.isaaclab.scenario_loader import build_table_texture_registry

    registry = build_table_texture_registry()
    if texture_key not in registry:
        raise KeyError(
            f"Unknown table texture key: {texture_key}. Available: {sorted(registry)}"
        )
    texture_path = registry[texture_key]
    before = collect_table_texture_debug(env)
    if log is not None:
        log(f"apply_table_texture: texture_key={texture_key}")
        log(f"apply_table_texture: texture_path={texture_path}")
        log(
            "apply_table_texture: table_prim_paths="
            + json.dumps(before["table_prim_paths"], ensure_ascii=False)
        )
    franka_stack_events.apply_visual_texture_material(
        env=env,
        env_ids=torch.tensor([0], device=env.device),
        asset_cfg=SceneEntityCfg("table"),
        texture=texture_path,
    )
    after = collect_table_texture_debug(env)
    report = {
        "texture_key": texture_key,
        "texture_path": texture_path,
        "before": before,
        "after": after,
    }
    if debug:
        print("[scenario debug] requested texture_key:", texture_key, flush=True)
        print("[scenario debug] resolved texture_path:", texture_path, flush=True)
        print(
            "[scenario debug] table prim paths:",
            before["table_prim_paths"],
            flush=True,
        )
        for shader_info in after["shader_paths"]:
            print("[scenario debug] shader:", shader_info, flush=True)
    if log is not None:
        for shader_info in after["shader_paths"]:
            log(
                "apply_table_texture: shader="
                + json.dumps(shader_info, ensure_ascii=False)
            )
        for proto_info in after.get("table_prototypes", []):
            log(
                "apply_table_texture: prototype="
                + json.dumps(proto_info, ensure_ascii=False)
            )
    return report


def main():
    args = parse_args()
    scenario = load_scenario(Path(args.scenario_file), args.scenario_id)
    output_path = Path(args.output)
    log_path = output_path.with_suffix(".debug.log")
    log = make_logger(log_path)
    log(f"start scenario render: scenario_id={args.scenario_id}")
    log(f"scenario file: {args.scenario_file}")
    log(f"output png: {output_path}")
    log("scenario record: " + json.dumps(scenario, ensure_ascii=False))

    from isaaclab.app import AppLauncher

    log("creating AppLauncher")
    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import load_cfg_from_registry

        log(f"loading env cfg from registry: env_id={args.env_id}")
        env_cfg = load_cfg_from_registry(args.env_id, "env_cfg_entry_point")
        env_cfg.seed = 0
        env_cfg.scene.num_envs = 1
        env_cfg.scene.wrist_cam.height = args.height
        env_cfg.scene.wrist_cam.width = args.width
        env_cfg.scene.table_cam.height = args.height
        env_cfg.scene.table_cam.width = args.width
        from rlinf.envs.isaaclab.scenario_loader import resolve_table_asset_path

        table_asset = scenario["table_asset"]
        table_asset_path = resolve_table_asset_path(table_asset, must_exist=True)
        env_cfg.scene.table.spawn.usd_path = table_asset_path
        log(f"scenario_id: {args.scenario_id}")
        log(f"table_asset: {table_asset}")
        log(f"resolved table usd_path: {table_asset_path}")

        log("creating gym env")
        env = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
        log("calling env.reset()")
        env.reset()
        log("env.reset() finished")

        apply_cube_pose(env, 0, "cube_1", scenario["cube_1_pos"], scenario["cube_1_rpy"])
        apply_cube_pose(env, 0, "cube_2", scenario["cube_2_pos"], scenario["cube_2_rpy"])
        apply_cube_pose(env, 0, "cube_3", scenario["cube_3_pos"], scenario["cube_3_rpy"])
        log("apply_cube_pose() finished for cube_1/2/3")

        for _ in range(4):
            env.sim.render()
        log("env.sim.render() x4 finished")
        obs = env.observation_manager.compute(update_history=True)
        log("observation_manager.compute() finished")
        image = obs["policy"]["table_cam"][0].detach().cpu().numpy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, image.astype("uint8"))
        log(f"saved preview png: {output_path}")
        if args.debug:
            debug_path = output_path.with_suffix(".debug.json")
            debug_payload = {
                "scenario_id": str(args.scenario_id),
                "scenario_record": scenario,
                "table_asset": table_asset,
                "resolved_table_asset_path": table_asset_path,
            }
            debug_path.write_text(
                json.dumps(debug_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log(f"saved debug json: {debug_path}")
            print(f"Saved scenario debug report to: {debug_path}")
        print(f"Saved scenario preview to: {output_path}")
    except Exception:
        log("exception during scenario render:")
        log(traceback.format_exc())
        raise
    finally:
        log("closing app")
        app.close()


if __name__ == "__main__":
    main()
