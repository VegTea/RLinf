# Copyright 2026 The RLinf Authors.
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

"""Tests for exporting RLinf pi05_droid SFT checkpoints."""

import importlib.util
from pathlib import Path

import pytest
import safetensors.torch
import torch

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "toolkits/standalone_eval_scripts/openpi/export_pi05_droid_sft_checkpoint.py"
)
_SPEC = importlib.util.spec_from_file_location("export_pi05_droid_sft", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_export_checkpoint_strips_wrapper_prefixes_and_matches_reference(tmp_path):
    checkpoint = tmp_path / "global_step_10"
    weights_dir = checkpoint / "actor/model_state_dict"
    weights_dir.mkdir(parents=True)
    trained = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save(
        {"_fsdp_wrapped_module.model.paligemma.weight": trained},
        weights_dir / "full_weights.pt",
    )

    reference = tmp_path / "reference"
    reference.mkdir()
    safetensors.torch.save_file(
        {"paligemma.weight": torch.zeros(2, 3, dtype=torch.float32)},
        reference / "model.safetensors",
    )
    (reference / "config.json").write_text('{"pi05": true}\n')
    norm_stats = tmp_path / "wipe_board"
    norm_stats.mkdir()
    (norm_stats / "norm_stats.json").write_text('{"norm_stats": {}}\n')

    output = tmp_path / "exported"
    output_path = _MODULE.export_checkpoint(checkpoint, reference, output, norm_stats)
    exported = safetensors.torch.load_file(str(output_path))

    assert exported["paligemma.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(exported["paligemma.weight"].float(), trained)
    assert (output / "config.json").read_text() == '{"pi05": true}\n'
    assert (output / "assets/wipe_board/norm_stats.json").read_text() == (
        '{"norm_stats": {}}\n'
    )


def test_resolve_full_weights_accepts_actor_directory(tmp_path):
    actor = tmp_path / "actor"
    path = actor / "model_state_dict/full_weights.pt"
    path.parent.mkdir(parents=True)
    path.touch()

    assert _MODULE.resolve_full_weights(actor) == path


def test_remove_redundant_tied_weights_accepts_equal_alias():
    canonical = "paligemma_with_expert.paligemma.lm_head.weight"
    alias = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    tensor = torch.arange(6).reshape(2, 3)
    trained = {canonical: tensor.clone(), alias: tensor.clone()}

    _MODULE.remove_redundant_tied_weights(trained, {canonical})

    assert set(trained) == {canonical}


def test_remove_redundant_tied_weights_rejects_different_alias():
    canonical = "paligemma_with_expert.paligemma.lm_head.weight"
    alias = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    trained = {canonical: torch.zeros(2, 3), alias: torch.ones(2, 3)}

    with pytest.raises(RuntimeError, match="differs from canonical tensor"):
        _MODULE.remove_redundant_tied_weights(trained, {canonical})
