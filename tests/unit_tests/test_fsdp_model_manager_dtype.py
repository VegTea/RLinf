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

"""Tests for FSDP model precision normalization."""

import torch
from torch import nn

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    _cast_model_to_configured_dtype,
)
from rlinf.hybrid_engines.fsdp.utils import init_fn


class _MixedDtypeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fp32 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.bf16 = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.register_buffer("bf16_buffer", torch.ones(2, dtype=torch.bfloat16))


class _ModelIgnoringModuleApply(_MixedDtypeModel):
    def _apply(self, fn, recurse=True):
        del fn, recurse
        return self


def test_explicit_precision_normalizes_custom_model_parameters_and_buffers():
    """An explicit model precision must override mixed checkpoint dtypes."""
    model = _cast_model_to_configured_dtype(_MixedDtypeModel(), torch.float32)

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert model.bf16_buffer.dtype == torch.float32


def test_native_checkpoint_precision_remains_unchanged_when_unspecified():
    """A null model precision must preserve native checkpoint dtypes."""
    model = _cast_model_to_configured_dtype(_MixedDtypeModel(), None)

    assert {parameter.dtype for parameter in model.parameters()} == {
        torch.float32,
        torch.bfloat16,
    }
    assert model.bf16_buffer.dtype == torch.bfloat16


def test_explicit_precision_handles_models_overriding_module_apply():
    """Registered parameters are normalized even if a custom model ignores to()."""
    model = _cast_model_to_configured_dtype(_ModelIgnoringModuleApply(), torch.float32)

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_fsdp_param_init_preserves_configured_dtype(monkeypatch):
    """FSDP materialization must not reintroduce mixed parameter precision."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    module = nn.Linear(2, 2, dtype=torch.float32)

    initialized = init_fn(module, dtype=torch.bfloat16)

    assert {parameter.dtype for parameter in initialized.parameters()} == {
        torch.bfloat16
    }
