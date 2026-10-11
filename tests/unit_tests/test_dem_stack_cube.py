# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests; dummy policy tests are NOT real DEM training evidence."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from rlinf.data.datasets.dem_stack_cube import StackCubeDataset
from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv
from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.models.embodiment.dem import DEMPolicy, get_model
from rlinf.models.embodiment.dem.stack_cube_io import (
    environment_state_to_dataset,
    prepare_stack_actions,
)


def test_rotation_roundtrip_and_signed_gripper():
    euler = np.array(
        [[3.140074, 0.066241, -3.123242], [-3.141006, 0.066199, -3.123242], [0, 0, 0]]
    )
    states = torch.zeros(3, 8)
    # Deliberately wrong legacy angles must never influence the DEM input.
    states[:, 3:6] = 123
    states[:, 6:] = torch.tensor([0.04, -0.04])
    quat = torch.tensor(Rotation.from_euler("xyz", euler).as_quat())
    restored = environment_state_to_dataset(states, quat, encoding="native_xyzw")
    error = (
        Rotation.from_euler("xyz", restored[:, 3:6])
        * Rotation.from_euler("xyz", euler).inv()
    ).magnitude()
    assert error.max() < 1e-6
    assert torch.equal(states[:, 6:], restored[:, 6:])
    candidate = environment_state_to_dataset(
        states, quat, encoding="legacy_wxyz_as_xyzw"
    )
    expected = Rotation.from_quat(quat.numpy()[:, [3, 0, 1, 2]])
    assert (
        Rotation.from_euler("xyz", candidate[:, 3:6]) * expected.inv()
    ).magnitude().max() < 1e-6
    for invalid in (torch.zeros(3, 4), quat * float("nan"), quat * 2, quat[:1]):
        with pytest.raises(ValueError):
            environment_state_to_dataset(states, invalid, encoding="native_xyzw")
    with pytest.raises(ValueError):
        environment_state_to_dataset(torch.zeros(3, 7), quat, encoding="native_xyzw")
    with pytest.raises(ValueError):
        environment_state_to_dataset(states, quat, encoding="guess")


def test_native_pose_opt_in_preserves_pi05_and_reaches_dem(monkeypatch):
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv
    from rlinf.models.embodiment.dem.stack_cube_io import INSTRUCTION

    env = object.__new__(IsaaclabStackCubeEnv)
    env.cfg = OmegaConf.create({})
    env.num_envs, env.task_description = 1, INSTRUCTION
    env.enable_depth_observations = False
    raw = {
        "policy": {
            "eef_pos": torch.tensor([[0.4, 0.0, 0.2]]),
            "eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            "gripper_pos": torch.tensor([[0.04, -0.04]]),
            "wrist_cam": torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
            "table_cam": torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
        }
    }
    old = env._wrap_obs(raw)
    assert "eef_quat_xyzw" not in old
    env.cfg.include_native_eef_quat = True
    new = env._wrap_obs(raw)
    assert torch.equal(old["states"], new["states"])
    assert torch.equal(new["eef_quat_xyzw"], raw["policy"]["eef_quat"])
    assert new["eef_quat_xyzw"].data_ptr() != raw["policy"]["eef_quat"].data_ptr()

    class Dummy(torch.nn.Module):
        device = torch.device("cpu")
        config = SimpleNamespace(chunk=16)

        def encode_vision(self, images):
            return torch.zeros(1, 2, 1, 1)

        def act(self, images, states, **kwargs):
            self.seen_states = states
            return torch.zeros(1, 16, 7)

    monkeypatch.setitem(
        sys.modules,
        "dem",
        SimpleNamespace(preprocess_images=lambda images, **kwargs: images),
    )
    policy = Dummy()
    model = DEMPolicy(policy, 5, 42, "native_xyzw")
    transmitted = EnvOutput(obs=new, final_obs=new).to_dict()
    assert torch.equal(transmitted["final_obs"]["eef_quat_xyzw"], new["eef_quat_xyzw"])
    merged = EnvOutput.merge_env_outputs([transmitted, transmitted])
    assert merged["obs"]["eef_quat_xyzw"].shape == (2, 4)
    assert "eef_quat_xyzw" not in EnvOutput(obs=old).to_dict()["obs"]
    actions, _ = model.predict_action_batch(transmitted["obs"])
    assert torch.equal(policy.seen_states[:, 3:6], torch.zeros(1, 3))
    assert not torch.equal(policy.seen_states[:, 3:6], old["states"][:, 3:6])
    assert actions.shape == (1, 5, 7) and (actions[..., -1] == 1).all()
    with pytest.raises(ValueError, match="native eef_quat_xyzw"):
        model.predict_action_batch(old)


def test_action_prefix_gripper_zero_and_no_rescaling():
    actions = torch.ones(2, 16, 7) * 0.02
    actions[0, :3, -1] = torch.tensor([-0.1, 0, 0.1])
    result = prepare_stack_actions(actions, 5)
    assert result.shape == (2, 5, 7)
    assert torch.equal(result[..., :6], actions[:, :5, :6])
    assert result[0, :3, -1].tolist() == [-1, 1, 1]
    assert actions[0, 1, -1] == 0
    with pytest.raises(ValueError):
        prepare_stack_actions(actions, 17)
    with pytest.raises(ValueError):
        prepare_stack_actions(actions * float("nan"), 5)


def test_dataset_boundaries_stats_and_camera_order(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps({"fps": 20, "codebase_version": "v2.1"})
    )
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    for episode in (0, 1):
        rows = {
            "observation.state": [[float(episode)] * 8] * 5,
            "action": [[float(episode * 10 + i)] * 7 for i in range(5)],
            "timestamp": [i / 20 for i in range(5)],
            "frame_index": list(range(5)),
            "episode_index": [episode] * 5,
        }
        pq.write_table(
            pa.Table.from_pydict(rows),
            tmp_path / f"data/chunk-000/episode_{episode:06d}.parquet",
        )
    data = StackCubeDataset(str(tmp_path), [0, 1], horizon=3)
    with patch.object(
        data,
        "frames",
        side_effect=lambda episode, camera: np.full(
            (5, 256, 256, 3), 11 if camera == "front" else 22, np.uint8
        ),
    ):
        assert len(data) == 6
        assert data[2]["actions"][:, 0].tolist() == [2, 3, 4]
        assert data[3]["actions"][:, 0].tolist() == [10, 11, 12]
        assert data[0]["images"][:, 0, 0, 0].tolist() == [11, 22]
    assert data.statistics()["action_mean"].tolist() == [7.0] * 7


def test_dem_factory_manifest_and_no_half_cast(tmp_path, monkeypatch):
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                chunk=16, proprio_dim=8, action_dim=7, num_cameras=2
            )

    path = tmp_path / "policy.pt"
    path.write_bytes(b"unit-test dummy; not DEM weights")
    manifest = {
        "state_rotation": "euler_xyz",
        "state_rotation_encoding": "native_xyzw",
        "gripper_state": "left_negative_right",
        "action_space": "isaaclab_raw_relative_ik_scale_0.5",
        "camera_keys": ["main_images", "wrist_images"],
        "fps": 20,
        "nfe": 1,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "contract_verified": True,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest))
    monkeypatch.setitem(
        sys.modules,
        "dem",
        SimpleNamespace(DEM=SimpleNamespace(load=lambda path: Dummy())),
    )
    cfg = OmegaConf.create(
        {"model_path": str(path), "num_action_chunks": 5, "dem": {"seed": 42}}
    )
    model = get_model(cfg)
    assert model.execute_steps == 5
    with pytest.raises(NotImplementedError):
        model.default_forward()
    with pytest.raises(ValueError, match="FP32"):
        get_model(cfg, torch.bfloat16)
    del manifest["state_rotation_encoding"]
    path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="state_rotation_encoding"):
        get_model(cfg)
    manifest["state_rotation_encoding"] = "native_xyzw"
    manifest["contract_verified"] = False
    path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract is unverified"):
        get_model(cfg)
    manifest["contract_verified"] = True
    path.with_suffix(".json").write_text(json.dumps(manifest))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        get_model(cfg)


def test_first_episode_metrics_chunk_mask_and_video_trim():
    env = object.__new__(IsaaclabBaseEnv)
    env.num_envs, env.device = 2, torch.device("cpu")
    env.worker_info = SimpleNamespace(rank=0)
    env.cfg = SimpleNamespace(max_episode_steps=4, first_episode_only=True)
    env.auto_reset, env.ignore_terminations = False, True
    env.render_only_at_chunk_end, env.trajectory_recorder = False, None
    env._episode_ids = torch.zeros(2, dtype=torch.int64)
    env._current_scenario_records = [None, None]
    env.init_cube_positions = None
    env.prev_step_reward = torch.zeros(2)
    env._elapsed_steps = torch.zeros(2, dtype=torch.int32)
    env._init_metrics()
    actions_seen = []
    transitions = iter(
        [
            ([0, 0], [True, False]),
            ([1, 1], [False, True]),
            ([0, 0], [False, False]),
            ([0, 0], [False, False]),
        ]
    )

    def step(actions, **kwargs):
        actions_seen.append(actions.clone())
        reward, done = next(transitions)
        return (
            None,
            torch.tensor(reward).float(),
            torch.tensor(done),
            torch.zeros(2).bool(),
            {},
        )

    env.env = SimpleNamespace(step=step)
    _, _, _, _, infos = env.chunk_step(torch.ones(2, 4, 7))
    result = infos[-1]["episode"]
    assert result["success_once"].tolist() == [False, True]
    assert result["success_once_window"].tolist() == [True, True]
    assert result["episode_len"].tolist() == [1, 2]
    assert result["first_success_step"].tolist() == [-1, 2]
    assert result["return"].tolist() == [0, 1]
    assert not actions_seen[1][0].any()
    assert not actions_seen[2].any()
    wrapper = RecordVideo(
        SimpleNamespace(seed=42, num_envs=2),
        OmegaConf.create(
            {"info_on_video": False, "per_env_videos": True, "first_episode_only": True}
        ),
    )
    images = [np.zeros((10, 10, 3), dtype=np.uint8)] * 2
    wrapper._append_frame(images, {}, None, None)
    for info in infos:
        wrapper._append_frame(images, info, None, None)
    # Never include internal-reset frames; terminal-image absence is explicit.
    assert list(map(len, wrapper.per_env_render_images)) == [1, 2]
    wrapper._executor.shutdown()
    env._reset_metrics(torch.tensor([0]))
    assert env._first_episode_end_step.tolist() == [0, 2]


def test_startup_physics_warmup_has_no_control_or_episode_step():
    from rlinf.envs.isaaclab.tasks.stack_cube import warmup_physics_before_reset

    events = []
    env = SimpleNamespace(
        physics_dt=0.01,
        sim=SimpleNamespace(step=lambda **kw: events.append(("physics", kw))),
        scene=SimpleNamespace(update=lambda **kw: events.append(("scene", kw))),
    )
    # There is deliberately no env.step/reset API in this fake.
    warmup_physics_before_reset(env, 0)
    assert not events
    warmup_physics_before_reset(env, 1)
    assert events == [("physics", {"render": False}), ("scene", {"dt": 0.01})]
    for invalid in (-1, 1.5, True, None):
        with pytest.raises(ValueError, match="nonnegative integer"):
            warmup_physics_before_reset(env, invalid)
    assert len(events) == 2


def test_eval_config_is_single_gpu_and_checkpoint_factory(monkeypatch):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    for name in (
        "RLINF_DEM_PYTHON",
        "RLINF_ISAACSIM_PYTHON",
        "REPO_PATH",
        "ISAACSIM_ASSET_ROOT",
        "DEM_POLICY_PATH",
    ):
        monkeypatch.setenv(name, "/test/" + name)
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="isaaclab_franka_stack_cube_eval_dem_isaacsim6")
    assert cfg.actor.model.model_type == "dem"
    assert cfg.runner.ckpt_path is None
    assert not cfg.actor.model.add_value_head
    assert (
        cfg.env.eval.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks
        == 0
    )
    assert str(cfg.cluster.component_placement.rollout.placement) == "0"
    assert str(cfg.cluster.component_placement.env.placement) == "0"
    assert cfg.env.eval.first_episode_only and not cfg.env.eval.auto_reset
    assert cfg.env.eval.include_native_eef_quat
    assert cfg.env.eval.init_params.startup_physics_warmup_steps == 1
