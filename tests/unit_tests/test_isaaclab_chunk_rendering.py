"""Unit checks for IsaacLab chunk-boundary camera rendering."""

from pathlib import Path
from types import MethodType

import torch
from omegaconf import OmegaConf

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv
from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv
from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_chunk_env(
    *, render_only_at_chunk_end: bool
) -> tuple[IsaaclabBaseEnv, list[bool], list[bool]]:
    env = object.__new__(IsaaclabBaseEnv)
    env.render_only_at_chunk_end = render_only_at_chunk_end
    env.auto_reset = False
    env.ignore_terminations = False
    env.device = torch.device("cpu")
    env.cfg = OmegaConf.create({})
    env.trajectory_recorder = None
    render_flags: list[bool] = []
    observation_flags: list[bool] = []

    def fake_step(
        self,
        actions=None,
        auto_reset=True,
        *,
        render_enabled=True,
        return_observation=True,
    ):
        del self, auto_reset
        render_flags.append(render_enabled)
        observation_flags.append(return_observation)
        num_envs = actions.shape[0]
        return (
            {"step": len(render_flags)} if return_observation else None,
            torch.zeros(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            torch.zeros(num_envs, dtype=torch.bool),
            {},
        )

    env.step = MethodType(fake_step, env)
    return env, render_flags, observation_flags


def test_chunk_optimization_only_renders_final_action() -> None:
    env, render_flags, observation_flags = _make_chunk_env(
        render_only_at_chunk_end=True
    )
    actions = torch.zeros(2, 5, 7)

    obs, rewards, terminations, truncations, infos = env.chunk_step(actions)

    assert render_flags == [False, False, False, False, True]
    assert observation_flags == [False, False, False, False, True]
    assert obs[:4] == [None, None, None, None]
    assert obs[-1] == {"step": 5}
    assert len(obs) == len(infos) == 5
    assert rewards.shape == terminations.shape == truncations.shape == (2, 5)


def test_legacy_chunk_mode_renders_every_action() -> None:
    env, render_flags, observation_flags = _make_chunk_env(
        render_only_at_chunk_end=False
    )

    env.chunk_step(torch.zeros(2, 5, 7))

    assert render_flags == [True, True, True, True, True]
    assert observation_flags == [True, True, True, True, True]


def test_stack_cube_rgb_only_observation_omits_depth() -> None:
    env = object.__new__(IsaaclabStackCubeEnv)
    env.task_description = "stack cubes"
    env.num_envs = 2
    env.enable_depth_observations = False
    policy_obs = {
        "table_cam": torch.zeros(2, 16, 16, 3, dtype=torch.uint8),
        "wrist_cam": torch.ones(2, 16, 16, 3, dtype=torch.uint8),
        "eef_pos": torch.zeros(2, 3),
        "eef_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "gripper_pos": torch.zeros(2, 1),
    }

    wrapped = env._wrap_obs({"policy": policy_obs})

    assert wrapped["main_images"] is policy_obs["table_cam"]
    assert wrapped["wrist_images"] is policy_obs["wrist_cam"]
    assert "main_images_depth" not in wrapped
    assert "wrist_images_depth" not in wrapped


def test_subprocess_step_transports_render_flag() -> None:
    class ActionQueue:
        payload = None

        def put(self, value):
            self.payload = value

    env = object.__new__(SubProcIsaacLabEnv)
    env.action_queue = ActionQueue()
    sent = []
    env._send_remote = lambda *values: sent.extend(values)
    env._get_obs_result = lambda context: context
    action = torch.zeros(2, 7)

    result = env.step(
        action, render_enabled=False, return_observation=False
    )

    queued_action, queued_render_flag, queued_observation_flag = env.action_queue.payload
    assert sent == ["step", "step"]
    assert queued_action is action
    assert queued_render_flag is False
    assert queued_observation_flag is False
    assert result == "step"


def test_optimized_configs_use_fast_train_and_full_cadence_eval() -> None:
    config_names = (
        "isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100.yaml",
        "isaaclab_franka_stack_cube_ppo_openpi_pi05_isaacsim6_chunkopt.yaml",
    )

    for config_name in config_names:
        cfg = OmegaConf.load(
            REPO_ROOT / "examples/embodiment/config" / config_name
        )

        assert cfg.env.train.init_params.enable_depth_observations is False
        assert cfg.env.train.init_params.render_only_at_chunk_end is True
        assert cfg.env.train.init_params.antialiasing_mode == "Off"
        assert cfg.env.eval.init_params.enable_depth_observations is False
        assert cfg.env.eval.init_params.render_only_at_chunk_end is False
        assert cfg.env.eval.init_params.antialiasing_mode == "DLAA"
