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

"""Record raw DROID deployment observations and model-input videos."""

from __future__ import annotations

import atexit
import datetime
import json
import logging
import os
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from openpi_client import base_policy
from openpi_client.image_tools import resize_with_pad
from typing_extensions import override

logger = logging.getLogger(__name__)


def _image_to_hwc_uint8(value: Any, key: str) -> np.ndarray:
    """Convert a deployment image to contiguous HWC uint8 RGB."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(
            f"Expected image {key!r} with 3 dimensions, got {image.shape}."
        )
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image {key!r}, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or np.nanmax(image) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _json_value(value: Any) -> Any:
    """Convert an observation value to a JSON-safe representation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class _H264Writer:
    """Lazy imageio-ffmpeg writer with browser/VS Code-compatible settings."""

    def __init__(self, path: pathlib.Path, fps: float) -> None:
        self._path = path
        self._fps = fps
        self._writer = None
        self._frame_size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        frame_size = (frame.shape[1], frame.shape[0])
        if self._writer is None:
            try:
                import imageio_ffmpeg
            except ImportError as exc:
                raise RuntimeError(
                    "imageio-ffmpeg is required for observation recording."
                ) from exc
            self._frame_size = frame_size
            self._writer = imageio_ffmpeg.write_frames(
                str(self._path),
                frame_size,
                fps=self._fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=1,
                output_params=["-movflags", "+faststart"],
                ffmpeg_log_level="error",
            )
            self._writer.send(None)
        elif frame_size != self._frame_size:
            raise ValueError(
                f"Video frame size changed for {self._path}: "
                f"expected {self._frame_size}, got {frame_size}."
            )
        self._writer.send(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class DroidObservationRecordingPolicy(base_policy.BasePolicy):
    """Record every raw request before forwarding it to a DROID policy."""

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        *,
        output_root: pathlib.Path,
        exterior_image_key: str,
        wrist_image_key: str = "observation/wrist_image_left",
        fps: float = 15.0,
    ) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {fps}.")
        self._policy = policy
        self._exterior_image_key = exterior_image_key
        self._wrist_image_key = wrist_image_key
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.output_dir = pathlib.Path(output_root).expanduser().resolve() / (
            f"session_{timestamp}_pid{os.getpid()}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._observations = (self.output_dir / "observations.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        self._writers = {
            "exterior": _H264Writer(self.output_dir / "exterior.mp4", fps),
            "wrist": _H264Writer(self.output_dir / "wrist.mp4", fps),
            "model_inputs": _H264Writer(self.output_dir / "model_inputs_224.mp4", fps),
        }
        self._index = 0
        self._closed = False
        (self.output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fps": fps,
                    "exterior_image_key": exterior_image_key,
                    "wrist_image_key": wrist_image_key,
                    "model_input_order": [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb_masked_zero",
                    ],
                    "process_id": os.getpid(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        atexit.register(self.close)
        logger.info("Recording DROID observations under %s", self.output_dir)

    def _record(self, obs: Mapping[str, Any]) -> None:
        timestamp_utc = datetime.datetime.now(datetime.UTC).isoformat()
        timestamp_unix_ns = time.time_ns()
        timestamp_monotonic_ns = time.monotonic_ns()
        missing = [
            key
            for key in (self._exterior_image_key, self._wrist_image_key)
            if key not in obs
        ]
        if missing:
            raise KeyError(f"Observation recording is missing image keys: {missing}.")

        exterior = _image_to_hwc_uint8(
            obs[self._exterior_image_key], self._exterior_image_key
        )
        wrist = _image_to_hwc_uint8(obs[self._wrist_image_key], self._wrist_image_key)
        exterior_224 = resize_with_pad(exterior, 224, 224)
        wrist_224 = resize_with_pad(wrist, 224, 224)
        model_inputs = np.concatenate(
            [exterior_224, wrist_224, np.zeros_like(exterior_224)], axis=1
        )
        self._writers["exterior"].write(exterior)
        self._writers["wrist"].write(wrist)
        self._writers["model_inputs"].write(model_inputs)

        image_keys = {
            key
            for key, value in obs.items()
            if isinstance(value, np.ndarray)
            and value.ndim == 3
            and (value.shape[0] == 3 or value.shape[-1] == 3)
        }
        record = {
            "index": self._index,
            "timestamp_utc": timestamp_utc,
            "timestamp_unix_ns": timestamp_unix_ns,
            "timestamp_monotonic_ns": timestamp_monotonic_ns,
            "observation": {
                key: (
                    {
                        "video": (
                            "exterior.mp4"
                            if key == self._exterior_image_key
                            else "wrist.mp4"
                            if key == self._wrist_image_key
                            else None
                        ),
                        "frame_index": self._index,
                        "shape": list(np.asarray(value).shape),
                        "dtype": str(np.asarray(value).dtype),
                    }
                    if key in image_keys
                    else _json_value(value)
                )
                for key, value in obs.items()
            },
        }
        self._observations.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._index += 1

    @override
    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        self._record(obs)
        return self._policy.infer(obs)

    @override
    def reset(self) -> None:
        self._policy.reset()

    def close(self) -> None:
        """Finalize all MP4 files and close the JSONL stream."""
        if self._closed:
            return
        self._closed = True
        for writer in self._writers.values():
            writer.close()
        self._observations.close()
        logger.info(
            "Finalized %d recorded observations in %s", self._index, self.output_dir
        )
