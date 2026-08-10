#!/usr/bin/env python3
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

"""Export an RLinf pi05_droid SFT checkpoint for OpenPI deployment."""

from __future__ import annotations

import argparse
import pathlib
import shutil
from collections.abc import Mapping

_WEIGHT_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)
_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_orig_mod.", "module.")
_TIED_WEIGHT_ALIASES = {
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": "paligemma_with_expert.paligemma.lm_head.weight",
}


def resolve_full_weights(checkpoint: pathlib.Path) -> pathlib.Path:
    """Resolve an RLinf consolidated checkpoint from common directory levels."""
    if checkpoint.is_file():
        return checkpoint
    for relative_path in _WEIGHT_CANDIDATES:
        candidate = checkpoint / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No full_weights.pt found under {checkpoint}; expected one of "
        f"{_WEIGHT_CANDIDATES}."
    )


def strip_fsdp_prefixes(state_dict: Mapping) -> dict:
    """Remove only FSDP/compile prefixes from an old-format OpenPI state dict."""
    result = {}
    for source_key, tensor in state_dict.items():
        key = source_key
        while True:
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            else:
                break
        if key in result:
            raise ValueError(f"Duplicate tensor key after prefix stripping: {key}")
        result[key] = tensor
    return result


def remove_redundant_tied_weights(trained: dict, reference_keys: set[str]) -> None:
    """Remove known tied-weight aliases omitted by the safetensors checkpoint."""
    import torch

    for alias, canonical in _TIED_WEIGHT_ALIASES.items():
        if alias not in trained or alias in reference_keys:
            continue
        if canonical not in trained or canonical not in reference_keys:
            raise RuntimeError(
                f"Cannot validate tied-weight alias {alias}: canonical tensor "
                f"{canonical} is unavailable."
            )
        alias_tensor = trained[alias]
        canonical_tensor = trained[canonical]
        if alias_tensor.shape != canonical_tensor.shape or not torch.equal(
            alias_tensor, canonical_tensor
        ):
            raise RuntimeError(
                f"Tied-weight alias {alias} differs from canonical tensor {canonical}."
            )
        del trained[alias]


def export_checkpoint(
    checkpoint: pathlib.Path,
    reference_checkpoint: pathlib.Path,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Validate and export RLinf weights as OpenPI ``model.safetensors``."""
    import safetensors.torch
    import torch

    weights_path = resolve_full_weights(checkpoint)
    reference_path = reference_checkpoint / "model.safetensors"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference checkpoint not found: {reference_path}")

    loaded = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(loaded, Mapping):
        raise TypeError(
            f"Expected a state-dict mapping in {weights_path}, got {type(loaded)}."
        )
    trained = strip_fsdp_prefixes(loaded)
    reference = safetensors.torch.load_file(str(reference_path), device="cpu")
    remove_redundant_tied_weights(trained, set(reference))

    missing = sorted(set(reference) - set(trained))
    extra = sorted(set(trained) - set(reference))
    if missing or extra:
        raise RuntimeError(
            "SFT/reference key mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)."
        )

    exported = {}
    for key, reference_tensor in reference.items():
        tensor = trained[key]
        if tuple(tensor.shape) != tuple(reference_tensor.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: trained={tuple(tensor.shape)}, "
                f"reference={tuple(reference_tensor.shape)}."
            )
        exported[key] = tensor.detach().cpu().to(reference_tensor.dtype).contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.safetensors"
    safetensors.torch.save_file(exported, str(output_path))
    config_path = reference_checkpoint / "config.json"
    if config_path.is_file():
        shutil.copy2(config_path, output_dir / "config.json")
    print(f"Exported {len(exported)} tensors from {weights_path} to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="RLinf global_step, actor, model_state_dict, or full_weights.pt path.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=pathlib.Path,
        required=True,
        help="Original pi05_droid PyTorch directory used to validate keys/shapes.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.reference_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
