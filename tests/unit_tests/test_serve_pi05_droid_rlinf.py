# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the RLinf pi0.5-DROID WebSocket serving helpers."""

import importlib.util
from pathlib import Path

import torch

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "toolkits"
    / "standalone_eval_scripts"
    / "openpi"
    / "serve_pi05_droid_rlinf.py"
)
_SPEC = importlib.util.spec_from_file_location("serve_pi05_droid_rlinf", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class _FakePi0:
    def __init__(self) -> None:
        self.grad_enabled = True
        self.image_shape = None

    def sample_actions(self, observation, *, num_steps, noise):
        del num_steps, noise
        self.grad_enabled = torch.is_grad_enabled()
        self.image_shape = observation.images["base_0_rgb"].shape
        return torch.zeros(1)


class _UpstreamObservation:
    images = {"base_0_rgb": torch.zeros(1, 3, 8, 12)}
    image_masks = {"base_0_rgb": torch.ones(1, dtype=torch.bool)}
    state = torch.zeros(1, 32)
    tokenized_prompt = torch.zeros(1, 4, dtype=torch.long)
    tokenized_prompt_mask = torch.ones(1, 4, dtype=torch.bool)
    token_ar_mask = None
    token_loss_mask = None


def test_adapted_sampler_disables_autograd():
    model = _FakePi0()
    _MODULE._adapt_sample_actions_for_openpi(model)

    output = model.sample_actions("cuda", _UpstreamObservation(), num_steps=10)

    assert not model.grad_enabled
    assert model.image_shape == (1, 8, 12, 3)
    assert output.shape == (1,)
