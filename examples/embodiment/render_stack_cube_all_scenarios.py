#!/usr/bin/env python

import argparse
import faulthandler
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import imageio


SCRIPT_PATH = Path(__file__).resolve()
RLINF_REPO_ROOT = SCRIPT_PATH.parents[2]
if str(RLINF_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_ROOT))

DEFAULT_ENV_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"
DEFAULT_SCENARIO_FILE = (
    RLINF_REPO_ROOT / "examples/embodiment/config/env/isaaclab_stack_cube.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render initial table_cam images for all stack-cube scenarios in a jsonl file."
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=DEFAULT_SCENARIO_FILE,
        help="Path to scenario jsonl file.",
    )
    parser.add_argument(
        "--scenario-ids",
        nargs="*",
        default=None,
        help="Optional subset of scenario ids. Defaults to all records in file order.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory under which the output folder is created. Defaults to a fresh logs/<YYYYMMDD-HH:MM:SS> directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output directory. Overrides --log-dir and default log discovery.",
    )
    parser.add_argument(
        "--output-name",
        default="scenario_initial_renders",
        help="Output subdirectory name when --output-dir is not given.",
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--render-frames", type=int, default=6)
    parser.add_argument("--save-wrist", action="store_true")
    parser.add_argument(
        "--watchdog-timeout",
        type=int,
        default=0,
        help=(
            "Dump all Python thread stacks to render.stack.log every N seconds. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--isolation",
        choices=("auto", "shared", "subprocess", "fresh-env", "app-reuse"),
        default="auto",
        help=(
            "Scenario isolation mode. auto uses subprocess when table_asset is present "
            "and shared otherwise."
        ),
    )
    parser.add_argument(
        "--fresh-env-per-scenario",
        action="store_true",
        help="Deprecated alias for --isolation fresh-env.",
    )
    parser.add_argument(
        "--child-render-one",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_scenarios(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise KeyError(f"Missing scenario id at {path}:{line_no}")
            records.append(record)
    if not records:
        raise ValueError(f"No scenario records found in {path}")
    return records


def select_scenarios(records, scenario_ids):
    if not scenario_ids:
        return records
    wanted = {str(scenario_id) for scenario_id in scenario_ids}
    selected = [record for record in records if str(record["id"]) in wanted]
    found = {str(record["id"]) for record in selected}
    missing = sorted(wanted - found)
    if missing:
        raise KeyError(f"Scenario ids not found: {missing}")
    return selected


def current_run_log_dir(log_root: Path):
    return log_root / time.strftime("%Y%m%d-%H:%M:%S", time.localtime())


def resolve_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir
    if args.log_dir is not None:
        return args.log_dir / args.output_name
    return current_run_log_dir(RLINF_REPO_ROOT / "logs") / args.output_name


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    return _log


def setup_stack_debugging(output_dir: Path, timeout: int, log):
    stack_path = output_dir / "render.stack.log"
    stack_fp = stack_path.open("a", encoding="utf-8")
    faulthandler.enable(file=stack_fp, all_threads=True)

    if timeout and timeout > 0:
        faulthandler.dump_traceback_later(
            timeout, repeat=True, file=stack_fp, exit=False
        )
        log(f"watchdog enabled timeout={timeout}s stack_log={stack_path}")
    else:
        log(f"stack dump log ready: {stack_path}")

    try:
        faulthandler.register(signal.SIGUSR1, file=stack_fp, all_threads=True)
        log("SIGUSR1 stack dump enabled")
    except (AttributeError, RuntimeError, ValueError) as exc:
        log(f"SIGUSR1 stack dump unavailable: {exc}")
    return stack_fp


def write_phase(output_dir: Path, scenario_id: str | None, phase: str, detail: str | None = None):
    payload = {
        "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "scenario_id": str(scenario_id) if scenario_id is not None else None,
        "phase": phase,
        "detail": detail,
    }
    (output_dir / "render.phase.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log_phase(output_dir: Path, log, scenario_id: str | None, phase: str, detail: str | None = None):
    suffix = f" detail={detail}" if detail else ""
    log(f"phase scenario_id={scenario_id} {phase}{suffix}")
    write_phase(output_dir, scenario_id, phase, detail)


def tensor_to_list(value):
    if value is None:
        return None
    return value.detach().cpu().tolist()


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

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    debug["world_translation"] = [float(translation[0]), float(translation[1]), float(translation[2])]
    debug["local_to_world_matrix"] = [
        [float(matrix[row][col]) for col in range(4)] for row in range(4)
    ]
    return debug


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


def configure_scenario_reset(env_cfg, scenario_file: Path, scenario_id: str):
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

    env_cfg.events.randomize_cube_positions.func = franka_stack_events.apply_scenario_reset
    env_cfg.events.randomize_cube_positions.params = {
        "asset_cfgs": [
            SceneEntityCfg("cube_1"),
            SceneEntityCfg("cube_2"),
            SceneEntityCfg("cube_3"),
        ],
        "scenario_file": str(scenario_file),
        "mode": "by_id",
        "loop": True,
        "fixed_ids": [str(scenario_id)],
        "external_group_a_ids": [],
        "external_group_b_ids": [],
        "ratio_a": 0.8,
        "worker_rank": 0,
        "total_workers": 1,
        "envs_per_worker": 1,
        "seed": 0,
    }


def clear_scenario_runtime_cache(env):
    for attr_name in (
        "_scenario_loader",
        "_scenario_scheduler",
        "_scenario_batch_index",
        "_scenario_bootstrap_consumed",
    ):
        if hasattr(env, attr_name):
            delattr(env, attr_name)


def scenario_has_table_asset(records):
    return any(record.get("table_asset") for record in records)


def resolve_isolation_mode(args, records):
    if args.fresh_env_per_scenario:
        return "fresh-env"
    if args.isolation != "auto":
        return args.isolation
    if scenario_has_table_asset(records):
        return "subprocess"
    return "shared"


def apply_preconstruction_table_asset(env_cfg, scenario, log):
    table_asset = scenario.get("table_asset")
    if not table_asset:
        return None

    from rlinf.envs.isaaclab.scenario_loader import resolve_table_asset_path

    table_asset_path = resolve_table_asset_path(table_asset, must_exist=True)
    env_cfg.scene.table.spawn.usd_path = table_asset_path
    log(f"preconstruction table_asset={table_asset} resolved={table_asset_path}")
    return table_asset_path


def apply_scenario_reset_direct(env, args, scenario):
    import torch
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

    scenario_id = str(scenario["id"])
    clear_scenario_runtime_cache(env)
    franka_stack_events.apply_scenario_reset(
        env=env,
        env_ids=torch.tensor([0], device=env.device, dtype=torch.long),
        asset_cfgs=[
            SceneEntityCfg("cube_1"),
            SceneEntityCfg("cube_2"),
            SceneEntityCfg("cube_3"),
        ],
        scenario_file=str(args.scenario_file),
        mode="by_id",
        loop=True,
        fixed_ids=[scenario_id],
        external_group_a_ids=[],
        external_group_b_ids=[],
        ratio_a=0.8,
        worker_rank=0,
        total_workers=1,
        envs_per_worker=1,
        seed=0,
    )


def render_current_env(env, args, scenario, output_dir: Path, log):
    scenario_id = str(scenario["id"])
    log_phase(output_dir, log, scenario_id, "render_current_env:start")
    table_cam = env.scene["table_cam"]
    table_cam_prim_path = camera_prim_path(table_cam, 0)
    table_cam_prim_after_reset = camera_prim_transform_debug(table_cam_prim_path)

    log_phase(output_dir, log, scenario_id, "sim.render:start", f"frames={max(int(args.render_frames), 1)}")
    log(f"render frames for scenario_id={scenario_id}")
    for frame_idx in range(max(int(args.render_frames), 1)):
        env.sim.render()
        log_phase(output_dir, log, scenario_id, "sim.render:frame_done", f"frame={frame_idx + 1}")
    if hasattr(table_cam, "reset"):
        log_phase(output_dir, log, scenario_id, "table_cam.reset:start")
        table_cam.reset([0])
        log_phase(output_dir, log, scenario_id, "table_cam.reset:done")
    log_phase(output_dir, log, scenario_id, "observation_compute:start")
    obs = env.observation_manager.compute(update_history=True)
    log_phase(output_dir, log, scenario_id, "observation_compute:done")

    table_png = output_dir / f"scenario_{scenario_id}_table.png"
    log_phase(output_dir, log, scenario_id, "write_table_png:start", str(table_png))
    imageio.imwrite(table_png, image_from_obs(obs, "table_cam"))
    log(f"saved {table_png}")
    log_phase(output_dir, log, scenario_id, "write_table_png:done", str(table_png))

    wrist_png = None
    if args.save_wrist:
        wrist_png = output_dir / f"scenario_{scenario_id}_wrist.png"
        imageio.imwrite(wrist_png, image_from_obs(obs, "wrist_cam"))
        log(f"saved {wrist_png}")

    if hasattr(table_cam, "_update_poses"):
        table_cam._update_poses([0])
    table_cam_prim_before_close = camera_prim_transform_debug(table_cam_prim_path)
    debug_payload = {
        "scenario_id": scenario_id,
        "scenario_record": scenario,
        "requested_table_asset": scenario.get("table_asset"),
        "requested_table_cam_pos": scenario.get("table_cam_pos"),
        "requested_table_cam_rot": scenario.get("table_cam_rot"),
        "scenario_reset_debug": getattr(env, "_scenario_last_table_cam_pose_by_env", {}).get(0),
        "scenario_table_asset_debug": getattr(env, "_scenario_last_table_asset_by_env", {}).get(0),
        "applied_table_cam_world_pos": tensor_to_list(table_cam.data.pos_w[0]),
        "applied_table_cam_world_rot_ros": tensor_to_list(table_cam.data.quat_w_ros[0]),
        "table_cam_prim_path": table_cam_prim_path,
        "table_cam_prim_transform_after_reset": table_cam_prim_after_reset,
        "table_cam_prim_transform_before_close": table_cam_prim_before_close,
        "table_png": str(table_png),
        "wrist_png": str(wrist_png) if wrist_png is not None else None,
    }
    debug_path = output_dir / f"scenario_{scenario_id}.debug.json"
    log_phase(output_dir, log, scenario_id, "write_debug_json:start", str(debug_path))
    debug_path.write_text(json.dumps(debug_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log_phase(output_dir, log, scenario_id, "render_current_env:done")
    return debug_payload


def render_one_scenario_shared_env(env, args, scenario, output_dir: Path, log):
    scenario_id = str(scenario["id"])
    log_phase(output_dir, log, scenario_id, "shared.env_reset:start")
    log(f"reset shared env for scenario_id={scenario_id}")
    env.reset()
    log_phase(output_dir, log, scenario_id, "shared.env_reset:done")
    log_phase(output_dir, log, scenario_id, "shared.apply_scenario_reset:start")
    log(f"apply scenario reset for scenario_id={scenario_id}")
    apply_scenario_reset_direct(env, args, scenario)
    log_phase(output_dir, log, scenario_id, "shared.apply_scenario_reset:done")
    return render_current_env(env, args, scenario, output_dir, log)


def settle_timeline_after_env_close(output_dir: Path, log, scenario_id: str, app, updates: int = 3):
    log_phase(output_dir, log, scenario_id, "timeline_settle:start")
    try:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        was_playing = bool(timeline.is_playing())
        log_phase(output_dir, log, scenario_id, "timeline_settle:state_before", f"is_playing={was_playing}")
        if was_playing:
            timeline.stop()
            if hasattr(timeline, "commit"):
                timeline.commit()
        for update_idx in range(max(int(updates), 0)):
            app.update()
            log_phase(output_dir, log, scenario_id, "timeline_settle:app_update", f"update={update_idx + 1}")
        log_phase(
            output_dir,
            log,
            scenario_id,
            "timeline_settle:state_after",
            f"is_playing={bool(timeline.is_playing())}",
        )
        log_phase(output_dir, log, scenario_id, "stage_reset:start")
        from isaaclab.sim.utils import stage as stage_utils

        stage_utils.create_new_stage()
        for update_idx in range(max(int(updates), 0)):
            app.update()
            log_phase(output_dir, log, scenario_id, "stage_reset:app_update", f"update={update_idx + 1}")
        log_phase(output_dir, log, scenario_id, "stage_reset:done")
    except Exception:
        log(f"timeline settle failed for scenario_id={scenario_id}")
        log(traceback.format_exc())
        raise
    finally:
        log_phase(output_dir, log, scenario_id, "timeline_settle:done")


def render_one_scenario_fresh_env(args, scenario, output_dir: Path, log, app=None):
    log_phase(output_dir, log, scenario.get("id"), "fresh.imports:start")
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry
    log_phase(output_dir, log, scenario.get("id"), "fresh.imports:done")

    scenario_id = str(scenario["id"])
    log_phase(output_dir, log, scenario_id, "load_cfg:start", args.env_id)
    env_cfg = load_cfg_from_registry(args.env_id, "env_cfg_entry_point")
    log_phase(output_dir, log, scenario_id, "load_cfg:done")
    log_phase(output_dir, log, scenario_id, "configure_env_cfg:start")
    configure_env_cfg(env_cfg, args)
    log_phase(output_dir, log, scenario_id, "configure_env_cfg:done")
    log_phase(output_dir, log, scenario_id, "preconstruction_table:start")
    apply_preconstruction_table_asset(env_cfg, scenario, log)
    log_phase(output_dir, log, scenario_id, "preconstruction_table:done")
    log_phase(output_dir, log, scenario_id, "configure_scenario_reset:start")
    configure_scenario_reset(env_cfg, args.scenario_file, scenario_id)
    log_phase(output_dir, log, scenario_id, "configure_scenario_reset:done")

    env = None
    try:
        log_phase(output_dir, log, scenario_id, "gym.make:start")
        log(f"create env for scenario_id={scenario_id}")
        env = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
        log_phase(output_dir, log, scenario_id, "gym.make:done")
        log_phase(output_dir, log, scenario_id, "env.reset:start")
        env.reset()
        log_phase(output_dir, log, scenario_id, "env.reset:done")
        return render_current_env(env, args, scenario, output_dir, log)
    finally:
        if env is not None:
            log_phase(output_dir, log, scenario_id, "env.close:start")
            env.close()
            log_phase(output_dir, log, scenario_id, "env.close:done")
            if app is not None and not args.child_render_one:
                settle_timeline_after_env_close(output_dir, log, scenario_id, app)


def subprocess_command(args, scenario, output_dir: Path):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--scenario-file",
        str(args.scenario_file),
        "--scenario-ids",
        str(scenario["id"]),
        "--output-dir",
        str(output_dir),
        "--output-name",
        str(args.output_name),
        "--env-id",
        str(args.env_id),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--render-frames",
        str(args.render_frames),
        "--isolation",
        "fresh-env",
        "--child-render-one",
    ]
    if args.watchdog_timeout:
        command.extend(["--watchdog-timeout", str(args.watchdog_timeout)])
    if args.save_wrist:
        command.append("--save-wrist")
    if args.debug:
        command.append("--debug")
    return command


def render_one_scenario_subprocess(args, scenario, output_dir: Path, log):
    scenario_id = str(scenario["id"])
    command = subprocess_command(args, scenario, output_dir)
    log_phase(output_dir, log, scenario_id, "subprocess.spawn:start", " ".join(command))
    log(f"spawn scenario process for scenario_id={scenario_id}")
    env = os.environ.copy()
    repo_path = str(RLINF_REPO_ROOT)
    env["PYTHONPATH"] = repo_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=str(RLINF_REPO_ROOT), env=env, check=True)
    log_phase(output_dir, log, scenario_id, "subprocess.spawn:done")
    debug_path = output_dir / f"scenario_{scenario_id}.debug.json"
    if not debug_path.exists():
        raise FileNotFoundError(f"Child process did not write debug json: {debug_path}")
    return json.loads(debug_path.read_text(encoding="utf-8"))


def render_scenarios_app_reuse(args, records, output_dir: Path, manifest_path: Path, log):
    from isaaclab.app import AppLauncher

    log_phase(output_dir, log, None, "app_reuse.AppLauncher:start")
    app = AppLauncher(headless=True, enable_cameras=True).app
    log_phase(output_dir, log, None, "app_reuse.AppLauncher:done")
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest_fp:
            for idx, scenario in enumerate(records, 1):
                try:
                    log(
                        f"render {idx}/{len(records)} scenario_id={scenario['id']} "
                        "isolation=app-reuse"
                    )
                    payload = render_one_scenario_fresh_env(args, scenario, output_dir, log, app=app)
                    log_phase(output_dir, log, scenario.get("id"), "manifest.write:start")
                    manifest_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    manifest_fp.flush()
                    log_phase(output_dir, log, scenario.get("id"), "manifest.write:done")
                except Exception:
                    log(f"failed scenario_id={scenario['id']}")
                    log(traceback.format_exc())
                    raise
    finally:
        log_phase(output_dir, log, None, "app_reuse.app.close:start")
        log("closing app-reuse app")
        app.close()
        log_phase(output_dir, log, None, "app_reuse.app.close:done")


def run_child_render_one(args, records, output_dir: Path, log):
    if len(records) != 1:
        raise ValueError("--child-render-one requires exactly one selected scenario")

    from isaaclab.app import AppLauncher

    log_phase(output_dir, log, records[0].get("id"), "child.AppLauncher:start")
    app = AppLauncher(headless=True, enable_cameras=True).app
    log_phase(output_dir, log, records[0].get("id"), "child.AppLauncher:done")
    try:
        payload = render_one_scenario_fresh_env(args, records[0], output_dir, log)
        manifest_path = output_dir / f"scenario_{records[0]['id']}.manifest.jsonl"
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        return payload
    finally:
        log_phase(output_dir, log, records[0].get("id"), "child.app.close:start")
        log("closing child app")
        app.close()
        log_phase(output_dir, log, records[0].get("id"), "child.app.close:done")


def main():
    stack_fp = None
    args = parse_args()
    records = select_scenarios(load_scenarios(args.scenario_file), args.scenario_ids)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(output_dir / "render.log")
    stack_fp = setup_stack_debugging(output_dir, args.watchdog_timeout, log)
    write_phase(output_dir, None, "main:start")
    isolation_mode = resolve_isolation_mode(args, records)
    log(f"scenario_file={args.scenario_file}")
    log(f"output_dir={output_dir}")
    log(f"scenario_count={len(records)}")
    log(f"isolation={isolation_mode}")
    if args.fresh_env_per_scenario:
        log("fresh_env_per_scenario=True mapped to isolation=fresh-env")

    from rlinf.envs.isaaclab.scenario_loader import resolve_table_asset_path

    for record in records:
        table_asset = record.get("table_asset")
        if table_asset:
            resolved = resolve_table_asset_path(table_asset, must_exist=True)
            log(f"scenario_id={record['id']} table_asset={table_asset} resolved={resolved}")

    if args.child_render_one:
        run_child_render_one(args, records, output_dir, log)
        print(f"Saved scenario initial render to: {output_dir}")
        return

    manifest_path = output_dir / "manifest.jsonl"

    if isolation_mode == "app-reuse":
        render_scenarios_app_reuse(args, records, output_dir, manifest_path, log)
        log(f"saved manifest={manifest_path}")
        print(f"Saved scenario initial renders to: {output_dir}")
        return

    if isolation_mode == "subprocess":
        with manifest_path.open("w", encoding="utf-8") as manifest_fp:
            for idx, scenario in enumerate(records, 1):
                try:
                    log(f"render {idx}/{len(records)} scenario_id={scenario['id']} isolation=subprocess")
                    payload = render_one_scenario_subprocess(args, scenario, output_dir, log)
                    manifest_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    manifest_fp.flush()
                except Exception:
                    log(f"failed scenario_id={scenario['id']}")
                    log(traceback.format_exc())
                    raise
        log(f"saved manifest={manifest_path}")
        print(f"Saved scenario initial renders to: {output_dir}")
        return

    log_phase(output_dir, log, None, "main.AppLauncher.import:start")
    from isaaclab.app import AppLauncher
    log_phase(output_dir, log, None, "main.AppLauncher.import:done")

    log_phase(output_dir, log, None, "main.AppLauncher:start")
    app = AppLauncher(headless=True, enable_cameras=True).app
    log_phase(output_dir, log, None, "main.AppLauncher:done")
    env = None
    try:
        if isolation_mode == "shared":
            import gymnasium as gym
            from isaaclab_tasks.utils import load_cfg_from_registry

            log("create shared IsaacLab env")
            env_cfg = load_cfg_from_registry(args.env_id, "env_cfg_entry_point")
            configure_env_cfg(env_cfg, args)
            env = gym.make(args.env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
            log("shared IsaacLab env created")
        elif isolation_mode != "fresh-env":
            raise ValueError(f"Unsupported isolation mode: {isolation_mode}")

        with manifest_path.open("w", encoding="utf-8") as manifest_fp:
            for idx, scenario in enumerate(records, 1):
                try:
                    log(f"render {idx}/{len(records)} scenario_id={scenario['id']} isolation={isolation_mode}")
                    if isolation_mode == "fresh-env":
                        payload = render_one_scenario_fresh_env(args, scenario, output_dir, log, app=app)
                    else:
                        payload = render_one_scenario_shared_env(env, args, scenario, output_dir, log)
                    manifest_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    manifest_fp.flush()
                except Exception:
                    log(f"failed scenario_id={scenario['id']}")
                    log(traceback.format_exc())
                    raise
        log(f"saved manifest={manifest_path}")
        print(f"Saved scenario initial renders to: {output_dir}")
    finally:
        if env is not None:
            try:
                log_phase(output_dir, log, None, "shared.env.close:start")
                log("closing shared env")
                env.close()
                log_phase(output_dir, log, None, "shared.env.close:done")
            except Exception:
                log("shared env.close() failed")
                log(traceback.format_exc())
        log_phase(output_dir, log, None, "main.app.close:start")
        log("closing app")
        app.close()
        log_phase(output_dir, log, None, "main.app.close:done")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
