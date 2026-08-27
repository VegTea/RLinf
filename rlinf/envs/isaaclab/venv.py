# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    parent_remote.close()
    env_fn = env_fn_wrapper.x
    isaac_env, sim_app = env_fn()
    device = isaac_env.device

    def _with_scenario_reset_info(reset_result):
        obs, info = reset_result
        info = dict(info) if isinstance(info, dict) else {}
        records = getattr(isaac_env, "_scenario_last_record_by_env", None)
        if isinstance(records, dict):
            info["scenario_records"] = dict(records)
        return obs, info

    def _to_tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _numpy_obs(obs):
        if isinstance(obs, torch.Tensor):
            return obs.detach().cpu().numpy()
        if isinstance(obs, dict):
            return {key: _numpy_obs(value) for key, value in obs.items()}
        if isinstance(obs, (list, tuple)):
            return [_numpy_obs(value) for value in obs]
        return obs

    def _apply_replay_state(payload):
        env_id = int(payload.get("env_id", 0))
        env_ids = torch.tensor([env_id], dtype=torch.long, device=device)

        robot = isaac_env.scene["robot"]
        joint_pos_rel = payload.get("joint_pos", None)
        if joint_pos_rel is not None:
            joint_pos_rel = _to_tensor(joint_pos_rel).reshape(1, -1)
            default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_pos = default_joint_pos + joint_pos_rel
            joint_vel = payload.get("joint_vel", None)
            if joint_vel is None:
                joint_vel = torch.zeros_like(joint_pos)
            else:
                joint_vel = _to_tensor(joint_vel).reshape(1, -1)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)

        object_state = payload.get("object", None)
        cube_positions = payload.get("cube_positions", None)
        cube_orientations = payload.get("cube_orientations", None)
        env_origin = isaac_env.scene.env_origins[env_id, :3].reshape(1, 3)
        for cube_idx, cube_name in enumerate(("cube_1", "cube_2", "cube_3")):
            cube = isaac_env.scene[cube_name]
            if object_state is not None:
                object_tensor = _to_tensor(object_state).reshape(-1)
                start = cube_idx * 7
                position = object_tensor[start : start + 3].reshape(1, 3) + env_origin
                orientation = object_tensor[start + 3 : start + 7].reshape(1, 4)
            elif cube_positions is not None and cube_orientations is not None:
                positions_tensor = _to_tensor(cube_positions).reshape(-1)
                orientations_tensor = _to_tensor(cube_orientations).reshape(-1)
                position = positions_tensor[cube_idx * 3 : cube_idx * 3 + 3].reshape(
                    1, 3
                )
                orientation = orientations_tensor[
                    cube_idx * 4 : cube_idx * 4 + 4
                ].reshape(1, 4)
            else:
                continue
            cube.write_root_pose_to_sim(
                torch.cat([position, orientation], dim=-1), env_ids=env_ids
            )
            cube.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=device), env_ids=env_ids
            )

        isaac_env.scene.write_data_to_sim()
        isaac_env.sim.render()
        isaac_env.scene.update(dt=0.0)
        obs = isaac_env.observation_manager.compute(update_history=False)
        return _numpy_obs(obs)

    def _get_prim_reference_asset_paths(prim) -> list[str]:
        asset_paths: list[str] = []
        try:
            prim_stack = prim.GetPrimStack()
        except Exception:
            return asset_paths
        for prim_spec in prim_stack:
            try:
                refs = prim_spec.referenceList.GetAddedOrExplicitItems()
            except Exception:
                continue
            for ref in refs:
                asset_path = getattr(ref, "assetPath", "")
                if asset_path:
                    asset_paths.append(str(asset_path))
        return asset_paths

    def _reference_matches_asset(reference_path: str, target_path: str) -> bool:
        try:
            return (
                reference_path == target_path
                or str(reference_path).split("/")[-1] == str(target_path).split("/")[-1]
            )
        except Exception:
            return False

    def _apply_replay_visual_scenario(payload):
        env_id = int(payload.get("env_id", 0))
        record = dict(payload.get("record", {}))
        applied: dict = {"env_id": env_id, "scenario_id": record.get("id")}

        table_asset_name = record.get("table_asset")
        if table_asset_name:
            from isaaclab.sim.utils.stage import get_current_stage

            from rlinf.envs.isaaclab.scenario_loader import resolve_table_asset_path

            table_asset_path = resolve_table_asset_path(
                table_asset_name, must_exist=True
            )
            table_asset = isaac_env.scene["table"]
            table_prim_path = str(table_asset.prim_paths[env_id])
            stage = get_current_stage()
            table_prim = stage.GetPrimAtPath(table_prim_path)
            if not table_prim or not table_prim.IsValid():
                raise RuntimeError(
                    f"Invalid table prim for replay visual scenario: {table_prim_path}"
                )
            previous_reference_paths = _get_prim_reference_asset_paths(table_prim)
            already_target_asset = any(
                _reference_matches_asset(reference_path, table_asset_path)
                for reference_path in previous_reference_paths
            )
            references = table_prim.GetReferences()
            if not already_target_asset:
                references.ClearReferences()
                references.AddReference(table_asset_path)
            applied["table_asset"] = {
                "requested": table_asset_name,
                "resolved": table_asset_path,
                "table_prim_path": table_prim_path,
                "previous_references": previous_reference_paths,
                "swapped": not already_target_asset,
                "active_references": _get_prim_reference_asset_paths(table_prim),
            }

        if "table_cam_pos" in record and "table_cam_rot" in record:
            table_cam = isaac_env.scene["table_cam"]
            local_pos = _to_tensor(record["table_cam_pos"]).reshape(3)
            ros_quat_wxyz = _to_tensor(record["table_cam_rot"]).reshape(4)
            ros_quat = ros_quat_wxyz[[1, 2, 3, 0]]
            world_pos = local_pos + isaac_env.scene.env_origins[env_id, 0:3]
            table_cam.set_world_poses(
                positions=world_pos.unsqueeze(0),
                orientations=ros_quat.unsqueeze(0),
                env_ids=[env_id],
                convention="ros",
            )
            if hasattr(table_cam, "_update_poses"):
                table_cam._update_poses([env_id])
            if hasattr(table_cam, "reset"):
                table_cam.reset([env_id])
            applied["table_cam"] = {
                "requested_local_pos": [float(v) for v in record["table_cam_pos"]],
                "requested_ros_quat_wxyz": [float(v) for v in record["table_cam_rot"]],
                "applied_ros_quat_xyzw": ros_quat.detach().cpu().tolist(),
                "applied_world_pos": world_pos.detach().cpu().tolist(),
                "applied_world_ros_quat": ros_quat.detach().cpu().tolist(),
            }

        isaac_env.sim.render()
        return applied

    def _set_replay_table_camera(payload):
        env_id = int(payload.get("env_id", 0))
        record = dict(payload.get("record", {}))
        if "table_cam_pos" not in record or "table_cam_rot" not in record:
            raise KeyError(
                "table camera record requires table_cam_pos and table_cam_rot"
            )
        table_cam = isaac_env.scene["table_cam"]
        local_pos = _to_tensor(record["table_cam_pos"]).reshape(3)
        ros_quat_wxyz = _to_tensor(record["table_cam_rot"]).reshape(4)
        ros_quat = ros_quat_wxyz[[1, 2, 3, 0]]
        world_pos = local_pos + isaac_env.scene.env_origins[env_id, 0:3]
        table_cam.set_world_poses(
            positions=world_pos.unsqueeze(0),
            orientations=ros_quat.unsqueeze(0),
            env_ids=[env_id],
            convention="ros",
        )
        if hasattr(table_cam, "_update_poses"):
            table_cam._update_poses([env_id])
        if hasattr(table_cam, "reset"):
            table_cam.reset([env_id])
        isaac_env.sim.render()
        return {
            "env_id": env_id,
            "scenario_id": record.get("id"),
            "requested_local_pos": [float(v) for v in record["table_cam_pos"]],
            "requested_ros_quat_wxyz": [float(v) for v in record["table_cam_rot"]],
            "applied_ros_quat_xyzw": ros_quat.detach().cpu().tolist(),
            "applied_world_pos": world_pos.detach().cpu().tolist(),
            "applied_world_ros_quat": ros_quat.detach().cpu().tolist(),
        }

    def _set_replay_external_cameras(payload):
        env_id = int(payload.get("env_id", 0))
        camera_names = list(payload.get("camera_names", []))
        records = list(payload.get("records", []))
        if len(camera_names) != len(records):
            raise ValueError(
                "replay external cameras require the same number of camera_names and records"
            )
        applied = []
        for camera_name, record in zip(camera_names, records, strict=False):
            record = dict(record)
            if "table_cam_pos" not in record or "table_cam_rot" not in record:
                raise KeyError(
                    f"camera record for {camera_name} requires table_cam_pos and table_cam_rot"
                )
            try:
                camera = isaac_env.scene[camera_name]
            except KeyError as exc:
                raise KeyError(
                    f"Replay camera is not registered in scene: {camera_name}"
                ) from exc
            local_pos = _to_tensor(record["table_cam_pos"]).reshape(3)
            ros_quat_wxyz = _to_tensor(record["table_cam_rot"]).reshape(4)
            ros_quat = ros_quat_wxyz[[1, 2, 3, 0]]
            world_pos = local_pos + isaac_env.scene.env_origins[env_id, 0:3]
            camera.set_world_poses(
                positions=world_pos.unsqueeze(0),
                orientations=ros_quat.unsqueeze(0),
                env_ids=[env_id],
                convention="ros",
            )
            if hasattr(camera, "_update_poses"):
                camera._update_poses([env_id])
            if hasattr(camera, "reset"):
                camera.reset([env_id])
            applied.append(
                {
                    "camera_name": str(camera_name),
                    "env_id": env_id,
                    "scenario_id": record.get("id"),
                    "requested_local_pos": [float(v) for v in record["table_cam_pos"]],
                    "requested_ros_quat_wxyz": [
                        float(v) for v in record["table_cam_rot"]
                    ],
                    "applied_ros_quat_xyzw": ros_quat.detach().cpu().tolist(),
                    "applied_world_pos": world_pos.detach().cpu().tolist(),
                    "applied_world_ros_quat": ros_quat.detach().cpu().tolist(),
                }
            )
        return {"env_id": env_id, "cameras": applied}

    def _get_replay_cube_color_map():
        color_to_cube: dict = {}
        cube_info: dict = {}
        try:
            from isaaclab.sim.utils.stage import get_current_stage
        except Exception as exc:
            return {
                "color_to_cube": color_to_cube,
                "cube_info": cube_info,
                "error": f"stage_import_failed: {exc}",
            }

        stage = get_current_stage()
        targets = {
            "red": torch.tensor([1.0, 0.0, 0.0], device=device),
            "green": torch.tensor([0.0, 1.0, 0.0], device=device),
            "blue": torch.tensor([0.0, 0.0, 1.0], device=device),
        }
        candidates = []
        for cube_name in ("cube_1", "cube_2", "cube_3"):
            info = {"prim_path": None, "diffuse_color": None, "material_paths": []}
            try:
                cube = isaac_env.scene[cube_name]
                prim_path = str(cube.prim_paths[0])
                info["prim_path"] = prim_path
                prim = stage.GetPrimAtPath(prim_path)
                color = _find_prim_diffuse_color(prim)
                if color is not None:
                    info["diffuse_color"] = [float(v) for v in color]
                    candidates.append(
                        (cube_name, torch.tensor(color[:3], device=device))
                    )
            except Exception as exc:
                info["error"] = str(exc)
            cube_info[cube_name] = info

        used_cubes = set()
        for color_name, target in targets.items():
            best_cube = None
            best_distance = None
            for cube_name, color in candidates:
                if cube_name in used_cubes:
                    continue
                distance = float(torch.linalg.norm(color - target).detach().cpu())
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_cube = cube_name
            if best_cube is not None:
                color_to_cube[color_name] = best_cube
                used_cubes.add(best_cube)
        return {"color_to_cube": color_to_cube, "cube_info": cube_info}

    def _apply_replay_light_scenario(payload):
        env_id = int(payload.get("env_id", 0))
        record = dict(payload.get("record", {}))
        light_cfg = dict(record.get("light", {}))
        if "intensity" not in light_cfg:
            raise KeyError(f"Missing light.intensity for id={record.get('id')}")
        if "color" not in light_cfg:
            raise KeyError(f"Missing light.color for id={record.get('id')}")

        intensity = float(light_cfg["intensity"])
        color = tuple(float(value) for value in light_cfg["color"])
        if len(color) != 3:
            raise ValueError(
                f"light.color must have 3 values for id={record.get('id')}"
            )
        texture = _resolve_replay_light_texture(light_cfg)

        light = isaac_env.scene["light"]
        light_prim = (
            light.prims[env_id] if len(light.prims) > env_id else light.prims[0]
        )
        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(color)
        light_prim.GetAttribute("inputs:texture:file").Set(texture)
        isaac_env.sim.render()

        return {
            "env_id": env_id,
            "id": str(record.get("id")),
            "preset": light_cfg.get("preset"),
            "requested_intensity": intensity,
            "requested_color": list(color),
            "requested_texture_key": light_cfg.get("texture_key", "default"),
            "requested_texture": light_cfg.get("texture", ""),
            "resolved_texture": texture,
            "light_prim_path": str(light_prim.GetPath()),
            "actual_intensity": _jsonable_attr_value(
                _attr_value(light_prim, "inputs:intensity")
            ),
            "actual_color": _jsonable_attr_value(
                _attr_value(light_prim, "inputs:color")
            ),
            "actual_texture": _jsonable_attr_value(
                _attr_value(light_prim, "inputs:texture:file")
            ),
        }

    def _resolve_replay_light_texture(light_cfg):
        explicit_texture = str(light_cfg.get("texture") or "")
        if explicit_texture:
            return explicit_texture
        key = str(light_cfg.get("texture_key", "default"))
        registry = _replay_light_texture_registry()
        if key not in registry:
            raise KeyError(f"Unknown texture_key={key}. Available: {sorted(registry)}")
        texture = registry[key]
        if texture:
            from pathlib import Path

            path = Path(texture)
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"DomeLight texture for texture_key={key} does not exist or is empty: {texture}"
                )
        return texture

    def _replay_light_texture_registry():
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        assets_root = repo_root.parent / "IsaacAssets5.1/Assets/Isaac/5.1"
        nvidia_root = assets_root / "NVIDIA"
        isaac_root = assets_root / "Isaac"
        return {
            "default": "",
            "abandoned_parking": str(
                nvidia_root
                / "Assets/AnimGraph/Worlds/textures/WarehouseInterior2b_4x8k.hdr"
            ),
            "evening_road": str(
                nvidia_root
                / "Assets/AnimGraph/105.0/Worlds/textures/adams_place_bridge_4k.hdr"
            ),
            "lakeside": str(
                isaac_root
                / "Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
            "autoshop": str(
                nvidia_root
                / "Assets/AnimGraph/Worlds/textures/ZetoCG_WarehouseInterior2b.hdr"
            ),
            "carpentry_shop": str(
                nvidia_root
                / "Assets/AnimGraph/Characters/Reallusion/Debra/Props/Stage/B2_2k_256_4k.hdr"
            ),
            "hospital_room": str(
                nvidia_root
                / "Assets/ArchVis/Industrial/Stages/HDRI/CloudyDay_04_04.hdr"
            ),
            "hotel_room": str(
                isaac_root
                / "Environments/Outdoor/Rivermark/dsready_content/nv_core/common_tools/content_tagging/studio_lights/Materials/photo_studio_01_4k.hdr"
            ),
            "small_empty_house": str(
                isaac_root
                / "Environments/Outdoor/Rivermark/dsready_content/nv_core/common_assets/environments/cloudy_sky/SubUSDs/textures/stars_4k.hdr"
            ),
            "photo_studio": str(
                isaac_root
                / "Environments/Outdoor/Rivermark/dsready_content/nv_core/common_tools/content_tagging/studio_lights/Materials/photo_studio_01_4k.hdr"
            ),
        }

    def _attr_value(prim, attr_name):
        attr = prim.GetAttribute(attr_name)
        return attr.Get() if attr else None

    def _jsonable_attr_value(value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        try:
            return [float(item) for item in value]
        except TypeError:
            return str(value)

    def _find_prim_diffuse_color(root_prim):
        if not root_prim or not root_prim.IsValid():
            return None
        stack = [root_prim]
        while stack:
            prim = stack.pop()
            for attr_name in (
                "inputs:diffuse_color",
                "inputs:diffuseColor",
                "diffuse_color",
                "diffuseColor",
                "displayColor",
            ):
                attr = prim.GetAttribute(attr_name)
                if not attr or not attr.IsValid():
                    continue
                try:
                    value = attr.Get()
                except Exception:
                    value = None
                if value is None:
                    continue
                try:
                    if len(value) > 0 and hasattr(value[0], "__len__"):
                        value = value[0]
                    return [float(value[0]), float(value[1]), float(value[2])]
                except Exception:
                    continue
            try:
                stack.extend(list(prim.GetChildren()))
            except Exception:
                pass
        return None

    try:
        while True:
            try:
                cmd = child_remote.recv()
            except EOFError:
                child_remote.close()
                break
            if cmd == "reset":
                reset_index, reset_seed = reset_idx_queue.get()
                if reset_index is None:
                    reset_result = isaac_env.reset(seed=reset_seed)
                else:
                    reset_result = isaac_env.reset(
                        seed=reset_seed, env_ids=reset_index.to(device)
                    )
                obs_queue.put(_with_scenario_reset_info(reset_result))
            elif cmd == "step":
                input_action = action_queue.get()
                step_result = isaac_env.step(input_action)
                obs_queue.put(step_result)
            elif cmd == "replay_set_state_and_get_obs":
                payload = child_remote.recv()
                obs_queue.put(_apply_replay_state(payload))
            elif cmd == "replay_apply_visual_scenario":
                payload = child_remote.recv()
                child_remote.send(_apply_replay_visual_scenario(payload))
            elif cmd == "replay_set_table_camera":
                payload = child_remote.recv()
                child_remote.send(_set_replay_table_camera(payload))
            elif cmd == "replay_set_external_cameras":
                payload = child_remote.recv()
                child_remote.send(_set_replay_external_cameras(payload))
            elif cmd == "replay_get_cube_color_map":
                child_remote.send(_get_replay_cube_color_map())
            elif cmd == "replay_apply_light_scenario":
                payload = child_remote.recv()
                child_remote.send(_apply_replay_light_scenario(payload))
            elif cmd == "close":
                isaac_env.close()
                child_remote.close()
                sim_app.close()
                break
            elif cmd == "device":
                child_remote.send(isaac_env.device)
            elif cmd == "set_scenario_curriculum_stage":
                stage_index = child_remote.recv()
                scheduler = getattr(isaac_env, "_scenario_scheduler", None)
                if scheduler is None:
                    child_remote.send(
                        {
                            "enabled": False,
                            "reason": "scenario_scheduler_not_initialized",
                        }
                    )
                else:
                    child_remote.send(scheduler.set_curriculum_stage(stage_index))
            elif cmd == "set_scenario_curriculum_progress":
                payload = child_remote.recv()
                scheduler = getattr(isaac_env, "_scenario_scheduler", None)
                if scheduler is None:
                    child_remote.send(
                        {
                            "enabled": False,
                            "reason": "scenario_scheduler_not_initialized",
                        }
                    )
                else:
                    child_remote.send(
                        scheduler.set_curriculum_progress(
                            payload.get("stage_index"),
                            payload.get("stage_step"),
                        )
                    )
            elif cmd == "get_scenario_curriculum_state":
                scheduler = getattr(isaac_env, "_scenario_scheduler", None)
                if scheduler is None:
                    child_remote.send(
                        {
                            "enabled": False,
                            "reason": "scenario_scheduler_not_initialized",
                        }
                    )
                else:
                    child_remote.send(scheduler.get_curriculum_state())
            else:
                child_remote.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        child_remote.close()
    finally:
        try:
            isaac_env.close()
        except Exception as e:
            print(f"IsaacLab Env Closed with error: {e}")


class SubProcIsaacLabEnv:
    def __init__(self, env_fn):
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.action_queue = ctx.Queue()
        self.obs_queue = ctx.Queue()
        self.reset_idx = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.obs_queue,
            self.reset_idx,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        self.isaac_lab_process.start()
        self.child_remote.close()

    def _get_obs_result(self, context: str, poll_interval: float = 1.0):
        while True:
            try:
                return self.obs_queue.get(timeout=poll_interval)
            except queue.Empty:
                if not self.isaac_lab_process.is_alive():
                    raise self._process_exit_error(context)

    def _process_exit_error(self, context: str) -> RuntimeError:
        if self.isaac_lab_process.exitcode is None:
            self.isaac_lab_process.join(timeout=0.5)
        return RuntimeError(
            "IsaacLab subprocess exited while waiting for "
            f"{context}; exitcode={self.isaac_lab_process.exitcode}"
        )

    def _send_remote(self, context: str, *values):
        try:
            for value in values:
                self.parent_remote.send(value)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise self._process_exit_error(context) from exc

    def _recv_remote(self, context: str, poll_interval: float = 1.0):
        while True:
            if self.parent_remote.poll(poll_interval):
                try:
                    return self.parent_remote.recv()
                except (BrokenPipeError, EOFError, OSError) as exc:
                    raise self._process_exit_error(context) from exc
            if not self.isaac_lab_process.is_alive():
                raise self._process_exit_error(context)

    def reset(self, seed=None, env_ids=None):
        self._send_remote("reset", "reset")
        self.reset_idx.put((env_ids, seed))
        obs, info = self._get_obs_result("reset")
        return obs, info

    def step(self, action: torch.Tensor):
        """
        action : (bs, action_dim)
        """
        self._send_remote("step", "step")
        self.action_queue.put(action)
        env_step_result = self._get_obs_result("step")
        return env_step_result

    def close(self):
        if self.isaac_lab_process.is_alive():
            try:
                self.parent_remote.send("close")
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.isaac_lab_process.join(timeout=5)
        if self.isaac_lab_process.is_alive():
            self.isaac_lab_process.terminate()
            self.isaac_lab_process.join(timeout=5)
        try:
            self.parent_remote.close()
        except OSError:
            pass

    def device(self):
        self._send_remote("device", "device")
        return self._recv_remote("device")

    def set_scenario_curriculum_stage(self, stage_index):
        self._send_remote(
            "set_scenario_curriculum_stage",
            "set_scenario_curriculum_stage",
            int(stage_index),
        )
        return self._recv_remote("set_scenario_curriculum_stage")

    def set_scenario_curriculum_progress(self, stage_index=None, stage_step=None):
        self._send_remote(
            "set_scenario_curriculum_progress",
            "set_scenario_curriculum_progress",
            {"stage_index": stage_index, "stage_step": stage_step},
        )
        return self._recv_remote("set_scenario_curriculum_progress")

    def get_scenario_curriculum_state(self):
        self._send_remote(
            "get_scenario_curriculum_state", "get_scenario_curriculum_state"
        )
        return self._recv_remote("get_scenario_curriculum_state")

    def replay_set_state_and_get_obs(self, payload: dict):
        self._send_remote(
            "replay_set_state_and_get_obs", "replay_set_state_and_get_obs", payload
        )
        return self._get_obs_result("replay_set_state_and_get_obs")

    def replay_apply_visual_scenario(self, payload: dict):
        self._send_remote(
            "replay_apply_visual_scenario", "replay_apply_visual_scenario", payload
        )
        return self._recv_remote("replay_apply_visual_scenario")

    def replay_set_table_camera(self, payload: dict):
        self._send_remote("replay_set_table_camera", "replay_set_table_camera", payload)
        return self._recv_remote("replay_set_table_camera")

    def replay_set_external_cameras(self, payload: dict):
        self._send_remote(
            "replay_set_external_cameras",
            "replay_set_external_cameras",
            payload,
        )
        return self._recv_remote("replay_set_external_cameras")

    def replay_get_cube_color_map(self):
        self._send_remote("replay_get_cube_color_map", "replay_get_cube_color_map")
        return self._recv_remote("replay_get_cube_color_map")

    def replay_apply_light_scenario(self, payload: dict):
        self._send_remote(
            "replay_apply_light_scenario", "replay_apply_light_scenario", payload
        )
        return self._recv_remote("replay_apply_light_scenario")
