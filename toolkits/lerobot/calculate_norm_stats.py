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

import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import pyarrow.parquet as pq
import tqdm
import tyro
from openpi.training.config import DataConfig

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.dataconfig.droid_dataconfig import (
    transform_pi05_droid_numeric_episode,
)


def calculate_pi05_droid_numeric_stats(
    dataset_root: pathlib.Path,
    *,
    action_horizon: int,
    action_dim: int,
    control_frequency_hz: float,
    batch_size: int,
) -> dict[str, normalize.NormStats]:
    """Compute pi0.5-DROID stats without decoding parquet-embedded images.

    This reproduces the LeRobot action query semantics used by OpenPI: each
    frame selects the current and future actions, clamps queries at the end of
    its episode, converts the seven absolute joint targets to joint velocities,
    and pads state/actions to the model action dimension.
    """
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No episode parquet files found under {dataset_root / 'data'}."
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    state_buffer: list[np.ndarray] = []
    action_buffer: list[np.ndarray] = []
    buffered_frames = 0

    def flush(*, force: bool = False) -> None:
        nonlocal state_buffer, action_buffer, buffered_frames
        if buffered_frames < batch_size and not force:
            return
        states = np.concatenate(state_buffer, axis=0)
        actions = np.concatenate(action_buffer, axis=0)
        offset = 0
        limit = len(states) if force else len(states) - (len(states) % batch_size)
        while offset < limit:
            end = min(offset + batch_size, limit)
            stats["state"].update(states[offset:end])
            stats["actions"].update(actions[offset:end])
            offset = end
        state_buffer = [states[offset:]] if offset < len(states) else []
        action_buffer = [actions[offset:]] if offset < len(actions) else []
        buffered_frames = len(states) - offset

    for parquet_path in tqdm.tqdm(parquet_paths, desc="Computing numeric stats"):
        table = pq.read_table(
            parquet_path,
            columns=["joint_position", "gripper_position", "actions"],
        )
        joint_positions = np.asarray(
            table["joint_position"].to_pylist(), dtype=np.float32
        )
        gripper_positions = np.asarray(
            table["gripper_position"].to_pylist(), dtype=np.float32
        ).reshape(-1, 1)
        absolute_actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        if joint_positions.shape != (len(table), 7):
            raise ValueError(
                f"Expected {parquet_path} joint_position shape ({len(table)}, 7), "
                f"got {joint_positions.shape}."
            )
        if absolute_actions.shape != (len(table), 8):
            raise ValueError(
                f"Expected {parquet_path} actions shape ({len(table)}, 8), "
                f"got {absolute_actions.shape}."
            )

        padded_states, padded_actions = transform_pi05_droid_numeric_episode(
            joint_positions,
            gripper_positions,
            absolute_actions,
            action_horizon=action_horizon,
            action_dim=action_dim,
            control_frequency_hz=control_frequency_hz,
        )
        state_buffer.append(padded_states)
        action_buffer.append(padded_actions)
        buffered_frames += len(table)
        flush()

    flush(force=True)
    return {key: value.get_statistics() for key, value in stats.items()}


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str,
    output_dir: str | None = None,
    numeric_only: bool = False,
):
    dataset_root = resolve_lerobot_dataset_root(repo_id)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset not found for repo_id={repo_id!r} at {dataset_root}. "
            "Pass a local dataset path, a Hugging Face repo id with data under "
            "HF_LEROBOT_HOME (default: ~/.cache/huggingface/lerobot), or download "
            "the dataset first."
        )
    config = get_openpi_config(
        config_name,
        repo_id=repo_id,
    )

    if numeric_only:
        if config_name != "pi05_droid":
            raise ValueError(
                "--numeric-only currently supports only --config-name pi05_droid."
            )
        norm_stats = calculate_pi05_droid_numeric_stats(
            dataset_root,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
            control_frequency_hz=config.data.control_frequency_hz,
            batch_size=config.batch_size,
        )
        output_path = (
            pathlib.Path(output_dir).expanduser()
            if output_dir is not None
            else config.assets_dirs / dataset_root.name
        )
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = (
        pathlib.Path(output_dir).expanduser()
        if output_dir is not None
        else config.assets_dirs / data_config.repo_id
    )
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
