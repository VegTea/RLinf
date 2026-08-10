# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for SFT-time decoding in the JAX-aligned OpenPI model."""

from types import SimpleNamespace

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)


class _FakePi0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_dim = 32
        self.received_observation = None
        self.received_noise = None

    def sample_actions(self, observation, *, num_steps, noise):
        self.received_observation = observation
        self.received_noise = noise
        assert num_steps == 10
        return noise + 1


def test_action_sample_adapts_upstream_observation_and_uses_fixed_noise() -> None:
    core = _FakePi0()
    model = OpenPiPytorchSFTActionModel(
        core,
        num_steps=10,
        action_env_dim=8,
    )
    upstream_observation = SimpleNamespace(
        images={"base_0_rgb": torch.zeros(2, 224, 224, 3)},
        image_masks={"base_0_rgb": torch.ones(2, dtype=torch.bool)},
        state=torch.zeros(2, 32),
        tokenized_prompt=torch.zeros(2, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )
    expert_actions = torch.zeros(2, 15, 32)
    fixed_noise = torch.randn(2, 15, 32)

    output = model(
        forward_type=ForwardType.ACTION_SAMPLE,
        data=(upstream_observation, expert_actions),
        noise=fixed_noise,
    )

    assert isinstance(core.received_observation, Observation)
    torch.testing.assert_close(core.received_noise, fixed_noise)
    torch.testing.assert_close(output, fixed_noise + 1)


def test_sft_rejects_unknown_forward_type() -> None:
    model = OpenPiPytorchSFTActionModel(
        _FakePi0(),
        num_steps=10,
        action_env_dim=8,
    )
    try:
        model(forward_type=ForwardType.DEFAULT)
    except NotImplementedError:
        return
    raise AssertionError("Unsupported forward type must raise NotImplementedError")
