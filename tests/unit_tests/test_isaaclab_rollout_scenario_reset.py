# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for inner versus outer scenario resets, without Isaac Sim."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from rlinf.envs.isaaclab.venv import _torch_worker


def test_inner_resets_reuse_scene_but_outer_resets_resample(monkeypatch):
    isaaclab = ModuleType("isaaclab")
    isaaclab.utils = ModuleType("isaaclab.utils")
    isaaclab.utils.math = ModuleType("isaaclab.utils.math")
    path = Path(__file__).resolve().parents[2] / "rlinf/envs/isaaclab/custom_events.py"
    spec = importlib.util.spec_from_file_location("scenario_events_test", path)
    events = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "isaaclab": isaaclab,
            "isaaclab.utils": isaaclab.utils,
            "isaaclab.utils.math": isaaclab.utils.math,
        },
    ):
        spec.loader.exec_module(events)
    scheduler = MagicMock()
    scheduler.sample_global_batch.side_effect = lambda size, index: [
        str(index * size + i) for i in range(size)
    ]
    loader = SimpleNamespace(get_by_ids=lambda ids: [{"id": i} for i in ids])
    monkeypatch.setattr(events, "_scenario_runtime", lambda *a: (loader, scheduler))

    def make_env():
        return SimpleNamespace(
            device="cpu",
            _scenario_bootstrap_consumed=False,
            _scenario_batch_index=0,
            _scenario_outer_reset=True,
        )

    def reset(env, ids=(0, 1), **options):
        events.apply_scenario_reset(
            env, torch.tensor(ids), [], "unused.jsonl", envs_per_worker=2, **options
        )
        return dict(env._scenario_last_record_by_env)

    env = make_env()
    reset(env, freeze_within_rollout=True)  # Initialization reset.
    initial = reset(env, freeze_within_rollout=True)  # First rollout epoch.
    calls = scheduler.sample_global_batch.call_count
    batch_index = env._scenario_batch_index
    env._scenario_outer_reset = False
    for ids in [(0,), (1,), (0, 1), (0,)]:
        assert reset(env, ids, freeze_within_rollout=True) == initial
    assert scheduler.sample_global_batch.call_count == calls
    assert env._scenario_batch_index == batch_index

    env._scenario_outer_reset = True
    changed = reset(env, (0,), freeze_within_rollout=True)
    assert changed[0] != initial[0]
    assert changed[1] == initial[1]  # Partial outer reset must not affect others.
    env._scenario_outer_reset = False
    assert reset(env, freeze_within_rollout=True) == changed
    env._scenario_outer_reset = True
    next_epoch = reset(env, freeze_within_rollout=True)
    assert all(next_epoch[i] != changed[i] for i in (0, 1))

    # Permanent evaluation freeze takes precedence, including on outer resets.
    fixed = make_env()
    initial = reset(fixed, freeze_per_env=True, freeze_within_rollout=True)
    calls = scheduler.sample_global_batch.call_count
    assert reset(fixed, freeze_per_env=True, freeze_within_rollout=True) == initial
    assert scheduler.sample_global_batch.call_count == calls

    # Default behavior remains resampling on internal resets.
    legacy = make_env()
    reset(legacy)
    reset(legacy)
    legacy._scenario_outer_reset = False
    calls = scheduler.sample_global_batch.call_count
    reset(legacy)
    assert scheduler.sample_global_batch.call_count == calls + 1


@pytest.mark.parametrize("ids", [None, torch.tensor([0])])
def test_worker_marks_unwrapped_outer_reset_and_clears_flag_on_error(ids):
    base = SimpleNamespace()

    def fail_reset(**kwargs):
        assert base._scenario_outer_reset is True
        assert kwargs["seed"] == 42
        if ids is not None:
            assert torch.equal(kwargs["env_ids"], ids)
        raise RuntimeError("reset probe")

    env = SimpleNamespace(
        device="cpu", unwrapped=base, reset=fail_reset, close=MagicMock()
    )
    child = MagicMock()
    child.recv.return_value = "reset"
    with pytest.raises(RuntimeError, match="reset probe"):
        _torch_worker(
            child,
            MagicMock(),
            SimpleNamespace(x=lambda: (env, None)),
            MagicMock(),
            MagicMock(),
            SimpleNamespace(get=lambda: (ids, 42)),
        )
    assert base._scenario_outer_reset is False
