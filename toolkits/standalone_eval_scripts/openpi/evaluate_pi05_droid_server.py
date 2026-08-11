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

"""Evaluate a pi05_droid WebSocket server on expert trajectory chunks."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

DEFAULT_DATASET_PATH = Path(
    "/inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "pi05_droid_eval"
EXTERIOR_CAMERA_KEYS = {
    "left": "observation/exterior_image_1_left",
    "right": "observation/exterior_image_2_left",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check a pi05_droid WebSocket server and compare predicted action "
            "chunks with expert chunks from one wipe-board episode."
        )
    )
    parser.add_argument(
        "--server-url",
        default="ws://127.0.0.1:8080",
        help="OpenPI WebSocket endpoint (default: ws://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"LeRobot dataset root (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=15)
    parser.add_argument("--control-frequency-hz", type=float, default=15.0)
    parser.add_argument(
        "--exterior-camera",
        choices=("auto", "left", "right"),
        default="auto",
        help="Exterior view to send. 'auto' uses the server metadata (default: auto).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the dataset task prompt for every request.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Connection and per-request timeout (default: 120).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Result directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def absolute_chunk_to_droid_actions(
    current_joint_position: np.ndarray,
    absolute_actions: np.ndarray,
    *,
    control_frequency_hz: float,
) -> np.ndarray:
    """Convert absolute targets to seven velocities plus absolute gripper."""
    current_joint_position = np.asarray(current_joint_position, dtype=np.float64)
    absolute_actions = np.asarray(absolute_actions, dtype=np.float64)
    if current_joint_position.shape != (7,):
        raise ValueError(
            "Expected current_joint_position with shape (7,), got "
            f"{current_joint_position.shape}."
        )
    if absolute_actions.ndim != 2 or absolute_actions.shape[1] < 8:
        raise ValueError(
            f"Expected absolute_actions with shape (horizon, >=8), got "
            f"{absolute_actions.shape}."
        )
    if not np.isfinite(control_frequency_hz) or control_frequency_hz <= 0:
        raise ValueError("control_frequency_hz must be finite and positive.")

    result = absolute_actions[:, :8].copy()
    previous_joint_positions = np.concatenate(
        [current_joint_position[None, :], absolute_actions[:-1, :7]], axis=0
    )
    result[:, :7] = (
        absolute_actions[:, :7] - previous_joint_positions
    ) * control_frequency_hz
    return result


def select_uniform_chunk_starts(
    episode_length: int,
    *,
    action_horizon: int,
    num_samples: int,
) -> np.ndarray:
    """Select evenly spaced starts whose chunks do not need end padding."""
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    valid_starts = episode_length - action_horizon + 1
    if valid_starts < num_samples:
        raise ValueError(
            f"Episode has only {valid_starts} complete chunk starts, fewer than "
            f"the requested {num_samples} samples."
        )
    return np.linspace(0, valid_starts - 1, num=num_samples, dtype=np.int64)


def compute_chunk_metrics(
    predicted_absolute: np.ndarray,
    expert_absolute: np.ndarray,
    current_joint_positions: np.ndarray,
    *,
    control_frequency_hz: float,
) -> dict[str, Any]:
    """Compute per-chunk and aggregate MSE in deployment and policy spaces."""
    predicted_absolute = np.asarray(predicted_absolute, dtype=np.float64)
    expert_absolute = np.asarray(expert_absolute, dtype=np.float64)
    current_joint_positions = np.asarray(current_joint_positions, dtype=np.float64)
    if predicted_absolute.shape != expert_absolute.shape:
        raise ValueError(
            "Predicted and expert chunks must have the same shape, got "
            f"{predicted_absolute.shape} and {expert_absolute.shape}."
        )
    if predicted_absolute.ndim != 3 or predicted_absolute.shape[-1] != 8:
        raise ValueError(
            "Expected action arrays shaped (samples, horizon, 8), got "
            f"{predicted_absolute.shape}."
        )
    if current_joint_positions.shape != (predicted_absolute.shape[0], 7):
        raise ValueError(
            "Current joint positions must be shaped (samples, 7), got "
            f"{current_joint_positions.shape}."
        )

    predicted_droid = np.stack(
        [
            absolute_chunk_to_droid_actions(
                current_joint_positions[index],
                predicted_absolute[index],
                control_frequency_hz=control_frequency_hz,
            )
            for index in range(len(predicted_absolute))
        ]
    )
    expert_droid = np.stack(
        [
            absolute_chunk_to_droid_actions(
                current_joint_positions[index],
                expert_absolute[index],
                control_frequency_hz=control_frequency_hz,
            )
            for index in range(len(expert_absolute))
        ]
    )

    absolute_squared_error = np.square(predicted_absolute - expert_absolute)
    droid_squared_error = np.square(predicted_droid - expert_droid)
    per_chunk = []
    for index in range(len(predicted_absolute)):
        per_chunk.append(
            {
                "absolute_position_mse": float(absolute_squared_error[index].mean()),
                "joint_position_mse": float(
                    absolute_squared_error[index, :, :7].mean()
                ),
                "droid_policy_space_mse": float(droid_squared_error[index].mean()),
                "joint_velocity_mse": float(droid_squared_error[index, :, :7].mean()),
                "gripper_mse": float(droid_squared_error[index, :, 7].mean()),
            }
        )

    return {
        "per_chunk": per_chunk,
        "aggregate": {
            "absolute_position_mse": float(absolute_squared_error.mean()),
            "joint_position_mse": float(absolute_squared_error[:, :, :7].mean()),
            "droid_policy_space_mse": float(droid_squared_error.mean()),
            "joint_velocity_mse": float(droid_squared_error[:, :, :7].mean()),
            "gripper_mse": float(droid_squared_error[:, :, 7].mean()),
        },
        "predicted_droid_actions": predicted_droid,
        "expert_droid_actions": expert_droid,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


class _TimedWebsocketPolicyClient:
    """Minimal OpenPI-compatible client with finite connection timeouts."""

    def __init__(self, uri: str, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._packer = msgpack_numpy.Packer()
        self._connection = connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_seconds,
            close_timeout=timeout_seconds,
        )
        metadata_message = self._connection.recv(timeout=timeout_seconds)
        if isinstance(metadata_message, str):
            raise RuntimeError(
                f"Expected binary server metadata, received: {metadata_message}"
            )
        self.metadata = msgpack_numpy.unpackb(metadata_message)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._connection.send(self._packer.pack(observation))
        response = self._connection.recv(timeout=self._timeout_seconds)
        if isinstance(response, str):
            raise RuntimeError(f"Error from inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._connection.close()


def _load_episode(dataset_path: Path, episode_index: int) -> Any:
    if not dataset_path.joinpath("meta", "info.json").is_file():
        raise FileNotFoundError(f"LeRobot metadata not found under {dataset_path}.")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative.")

    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(1)
    return LeRobotDataset(
        dataset_path.name,
        root=dataset_path,
        episodes=[episode_index],
        download_videos=False,
    )


def resolve_exterior_camera(
    requested: str, metadata: dict[str, Any]
) -> tuple[str, str]:
    """Resolve and validate the client exterior view against server metadata."""
    metadata_key = metadata.get("exterior_image_key")
    if requested == "auto":
        matches = [
            name for name, key in EXTERIOR_CAMERA_KEYS.items() if key == metadata_key
        ]
        if len(matches) != 1:
            raise ValueError(
                "Cannot infer exterior camera from server metadata: "
                f"exterior_image_key={metadata_key!r}."
            )
        requested = matches[0]
    image_key = EXTERIOR_CAMERA_KEYS[requested]
    if metadata_key is not None and metadata_key != image_key:
        raise ValueError(
            f"Requested {requested!r} exterior camera ({image_key}) but server "
            f"expects {metadata_key}."
        )
    return requested, image_key


def _make_observation(
    sample: dict[str, Any], prompt: str | None, exterior_camera: str
) -> dict[str, Any]:
    joint_position = _to_numpy(sample["joint_position"]).astype(np.float32)
    image_key = EXTERIOR_CAMERA_KEYS[exterior_camera]
    dataset_image_key = image_key.removeprefix("observation/")
    return {
        image_key: _to_numpy(sample[dataset_image_key]),
        "observation/wrist_image_left": _to_numpy(sample["wrist_image_left"]),
        "observation/joint_position": joint_position,
        "observation/gripper_position": _to_numpy(sample["gripper_position"]).astype(
            np.float32
        ),
        "prompt": prompt if prompt is not None else sample["task"],
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if not np.isfinite(args.control_frequency_hz) or args.control_frequency_hz <= 0:
        raise ValueError("--control-frequency-hz must be finite and positive.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset = _load_episode(args.dataset_path, args.episode_index)
    if float(dataset.fps) != args.control_frequency_hz:
        logging.warning(
            "Dataset FPS (%s) differs from evaluation control frequency (%s).",
            dataset.fps,
            args.control_frequency_hz,
        )

    sample_indices = select_uniform_chunk_starts(
        len(dataset),
        action_horizon=args.action_horizon,
        num_samples=args.num_samples,
    )
    all_actions = np.stack(
        [_to_numpy(action) for action in dataset.hf_dataset["actions"]]
    ).astype(np.float32)

    logging.info("Connecting to %s", args.server_url)
    client = _TimedWebsocketPolicyClient(args.server_url, args.timeout_seconds)
    logging.info("Server metadata: %s", client.metadata)
    try:
        exterior_camera, exterior_image_key = resolve_exterior_camera(
            args.exterior_camera, client.metadata
        )
    except Exception:
        client.close()
        raise
    logging.info("Using %s exterior camera (%s)", exterior_camera, exterior_image_key)

    predicted_chunks = []
    expert_chunks = []
    current_joint_positions = []
    frame_indices = []
    latencies_ms = []
    try:
        for sample_number, local_index in enumerate(sample_indices, start=1):
            sample = dataset[int(local_index)]
            observation = _make_observation(sample, args.prompt, exterior_camera)
            start_time = time.monotonic()
            result = client.infer(observation)
            latency_ms = (time.monotonic() - start_time) * 1000
            if "actions" not in result:
                raise KeyError("Server response does not contain 'actions'.")

            predicted = np.asarray(result["actions"], dtype=np.float32)
            if predicted.ndim != 2 or predicted.shape[0] < args.action_horizon:
                raise ValueError(
                    "Server returned an action chunk with shape "
                    f"{predicted.shape}; expected at least "
                    f"({args.action_horizon}, 8)."
                )
            predicted = predicted[: args.action_horizon, :8]
            expert = all_actions[
                int(local_index) : int(local_index) + args.action_horizon, :8
            ]
            if expert.shape != predicted.shape:
                raise ValueError(
                    f"Expert chunk shape {expert.shape} does not match prediction "
                    f"shape {predicted.shape}."
                )

            frame_index = int(_to_numpy(sample["frame_index"]).item())
            predicted_chunks.append(predicted)
            expert_chunks.append(expert)
            current_joint_positions.append(observation["observation/joint_position"])
            frame_indices.append(frame_index)
            latencies_ms.append(latency_ms)
            logging.info(
                "Inference %d/%d: local_index=%d frame_index=%d latency=%.1f ms",
                sample_number,
                args.num_samples,
                local_index,
                frame_index,
                latency_ms,
            )
    finally:
        client.close()

    predicted_chunks_array = np.stack(predicted_chunks)
    expert_chunks_array = np.stack(expert_chunks)
    current_joint_positions_array = np.stack(current_joint_positions)
    metrics = compute_chunk_metrics(
        predicted_chunks_array,
        expert_chunks_array,
        current_joint_positions_array,
        control_frequency_hz=args.control_frequency_hz,
    )

    rows = []
    for index, chunk_metrics in enumerate(metrics["per_chunk"]):
        row = {
            "sample": index,
            "local_index": int(sample_indices[index]),
            "frame_index": frame_indices[index],
            "latency_ms": latencies_ms[index],
            **chunk_metrics,
        }
        rows.append(row)
        print(
            f"sample={index:02d} frame={frame_indices[index]:04d} "
            f"absolute_mse={row['absolute_position_mse']:.8f} "
            f"droid_mse={row['droid_policy_space_mse']:.8f} "
            f"latency_ms={row['latency_ms']:.1f}"
        )

    aggregate = metrics["aggregate"]
    aggregate["mean_latency_ms"] = float(np.mean(latencies_ms))
    print("\nAggregate metrics")
    for key, value in aggregate.items():
        print(f"  {key}: {value:.8f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{exterior_camera}_episode_{args.episode_index:06d}_{args.num_samples}_chunks"
    )
    summary_path = args.output_dir / f"{stem}.json"
    chunks_path = args.output_dir / f"{stem}.npz"
    summary = {
        "server_url": args.server_url,
        "server_metadata": client.metadata,
        "dataset_path": str(args.dataset_path),
        "episode_index": args.episode_index,
        "num_samples": args.num_samples,
        "action_horizon": args.action_horizon,
        "control_frequency_hz": args.control_frequency_hz,
        "exterior_camera": exterior_camera,
        "exterior_image_key": exterior_image_key,
        "rows": rows,
        "aggregate": aggregate,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        chunks_path,
        sample_indices=sample_indices,
        frame_indices=np.asarray(frame_indices),
        current_joint_positions=current_joint_positions_array,
        predicted_absolute_actions=predicted_chunks_array,
        expert_absolute_actions=expert_chunks_array,
        predicted_droid_actions=metrics["predicted_droid_actions"],
        expert_droid_actions=metrics["expert_droid_actions"],
    )
    print(f"\nWrote summary to {summary_path}")
    print(f"Wrote action chunks to {chunks_path}")


if __name__ == "__main__":
    main()
