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

"""Load one transformed wipe-board batch through the official OpenPI loader."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Build the pi05_droid loader and print one batch's model input shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    args = parser.parse_args()

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config(
        "pi05_droid",
        model_path=str(args.checkpoint_dir),
        batch_size=1,
        repo_id=str(args.dataset_path),
        data_kwargs={
            "control_frequency_hz": 15.0,
            "exterior_image_key": "observation/exterior_image_2_left",
            "norm_stats_path": str(args.norm_stats_dir),
        },
    )
    loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=False
    )
    observation, actions = next(iter(loader))
    print(f"actions: shape={tuple(actions.shape)}, dtype={actions.dtype}")
    print(
        f"state: shape={tuple(observation.state.shape)}, "
        f"dtype={observation.state.dtype}"
    )
    print(
        "images: "
        + str({key: tuple(value.shape) for key, value in observation.images.items()})
    )
    print(
        "image_masks: "
        + str({key: bool(value[0]) for key, value in observation.image_masks.items()})
    )
    print(
        f"prompt: shape={tuple(observation.tokenized_prompt.shape)}, "
        f"dtype={observation.tokenized_prompt.dtype}"
    )


if __name__ == "__main__":
    main()
