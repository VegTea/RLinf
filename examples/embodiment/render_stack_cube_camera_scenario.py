import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import imageio


SCRIPT_PATH = Path(__file__).resolve()
RLINF_REPO_ROOT = SCRIPT_PATH.parents[2]
if str(RLINF_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one stack-cube scenario using the scenario reset path, including table_cam pose."
    )
    parser.add_argument("--scenario-file", type=str, required=True)
    parser.add_argument("--scenario-id", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--env-id",
        type=str,
        default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--render-frames", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
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


def tensor_to_list(value):
    if value is None:
        return None
    return value.detach().cpu().tolist()


def force_camera_observation_refresh(camera, env_id: int):
    # env.reset() computes observations once. Mark the camera dirty again so
    # debug renders read the latest RTX annotator buffer after manual renders.
    camera.reset([env_id])


def camera_prim_path(camera, env_id: int = 0):
    if hasattr(camera, "_view") and hasattr(camera._view, "prim_paths"):
        return str(camera._view.prim_paths[env_id])
    if hasattr(camera, "prim_paths"):
        return str(camera.prim_paths[env_id])
    return None


def camera_prim_transform_debug(prim_path: str | None):
    if prim_path is None:
        return None

    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    debug = {
        "prim_path": prim_path,
        "is_valid": bool(prim and prim.IsValid()),
    }
    if not prim or not prim.IsValid():
        return debug

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    debug["world_translation"] = [float(translation[0]), float(translation[1]), float(translation[2])]
    debug["local_to_world_matrix"] = [
        [float(matrix[row][col]) for col in range(4)] for row in range(4)
    ]
    return debug


def main():
    args = parse_args()
    scenario = load_scenario(Path(args.scenario_file), args.scenario_id)
    output_path = Path(args.output)
    log = make_logger(output_path.with_suffix(".debug.log"))
    log(f"start camera scenario render: scenario_id={args.scenario_id}")
    log("scenario record: " + json.dumps(scenario, ensure_ascii=False))

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

        env_cfg = load_cfg_from_registry(args.env_id, "env_cfg_entry_point")
        env_cfg.seed = 0
        env_cfg.scene.num_envs = 1
        env_cfg.scene.wrist_cam.height = args.height
        env_cfg.scene.wrist_cam.width = args.width
        env_cfg.scene.table_cam.height = args.height
        env_cfg.scene.table_cam.width = args.width

        env_cfg.events.randomize_cube_positions.func = (
            franka_stack_events.apply_scenario_reset
        )
        env_cfg.events.randomize_cube_positions.params = {
            "asset_cfgs": [
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ],
            "scenario_file": args.scenario_file,
            "mode": "by_id",
            "loop": True,
            "fixed_ids": [str(args.scenario_id)],
            "external_group_a_ids": [],
            "external_group_b_ids": [],
            "ratio_a": 0.8,
            "worker_rank": 0,
            "total_workers": 1,
            "envs_per_worker": 1,
            "seed": 0,
        }
        if hasattr(env_cfg.events, "randomize_table_visual_material"):
            env_cfg.events.randomize_table_visual_material.func = (
                franka_stack_events.noop_event
            )
            env_cfg.events.randomize_table_visual_material.params = {}

        env = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
        env.reset()
        table_cam = env.scene["table_cam"]
        table_cam_prim_path = camera_prim_path(table_cam, 0)
        prim_debug_after_reset = camera_prim_transform_debug(table_cam_prim_path)
        for _ in range(max(int(args.render_frames), 1)):
            env.sim.render()
        force_camera_observation_refresh(table_cam, 0)
        obs = env.observation_manager.compute(update_history=True)
        image = obs["policy"]["table_cam"][0].detach().cpu().numpy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, image.astype("uint8"))
        log(f"saved table_cam image: {output_path}")

        if hasattr(table_cam, "_update_poses"):
            table_cam._update_poses([0])
        prim_debug_before_close = camera_prim_transform_debug(table_cam_prim_path)
        debug_payload = {
            "scenario_id": str(args.scenario_id),
            "scenario_record": scenario,
            "requested_table_cam_pos": scenario.get("table_cam_pos"),
            "requested_table_cam_rot": scenario.get("table_cam_rot"),
            "scenario_reset_debug": getattr(
                env, "_scenario_last_table_cam_pose_by_env", {}
            ).get(0),
            "scenario_table_asset_debug": getattr(
                env, "_scenario_last_table_asset_by_env", {}
            ).get(0),
            "applied_table_cam_world_pos": tensor_to_list(table_cam.data.pos_w[0]),
            "applied_table_cam_world_rot_ros": tensor_to_list(table_cam.data.quat_w_ros[0]),
            "table_cam_prim_path": table_cam_prim_path,
            "table_cam_prim_transform_after_reset": prim_debug_after_reset,
            "table_cam_prim_transform_before_close": prim_debug_before_close,
            "output_png": str(output_path),
        }
        debug_path = output_path.with_suffix(".debug.json")
        debug_path.write_text(
            json.dumps(debug_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if args.debug:
            print(json.dumps(debug_payload, indent=2, ensure_ascii=False))
        log(f"saved debug json: {debug_path}")
        print(f"Saved camera scenario preview to: {output_path}")
    except Exception:
        log("exception during camera scenario render:")
        log(traceback.format_exc())
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                log("env.close() failed during cleanup")
                log(traceback.format_exc())
        log("closing app")
        app.close()


if __name__ == "__main__":
    main()
