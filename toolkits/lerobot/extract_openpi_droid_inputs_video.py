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

"""Export the three OpenPI pi05_droid image slots for one LeRobot episode.

The video panels, from left to right, are ``base_0_rgb``,
``left_wrist_0_rgb``, and ``right_wrist_0_rgb``. The official DROID policy
uses the first two inputs and masks the final all-zero padding slot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATASET_PATH = (
    "/inspire/hdd/global_user/czxs24230043/data/wipe_board_v1_zed196_force"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "extract_videos"
_MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_EXTERIOR_CAMERA_KEYS = {
    "left": "exterior_image_1_left",
    "right": "exterior_image_2_left",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write one MP4 whose three horizontal panels are the exact image slots "
            "provided to the official OpenPI pi05_droid model."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(DEFAULT_DATASET_PATH),
        help=f"LeRobot dataset root (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode index to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the MP4 (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate. Defaults to the dataset FPS.",
    )
    parser.add_argument(
        "--external-camera",
        choices=tuple(_EXTERIOR_CAMERA_KEYS),
        default="right",
        help=(
            "Exterior view mapped to base_0_rgb (default: right). "
            "left uses exterior_image_1_left; right uses exterior_image_2_left."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output video.",
    )
    return parser


def _require_imageio_ffmpeg() -> Any:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required to encode the H.264 MP4 output."
        ) from exc
    return imageio_ffmpeg


def _tensor_to_hwc_uint8(image: Any) -> np.ndarray:
    """Match ``openpi.policies.droid_policy._parse_image`` for LeRobot frames."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.ascontiguousarray(image.transpose(1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image data, got shape {image.shape}.")
    return image


def _get_pi05_droid_images(
    sample: dict[str, Any], exterior_camera_key: str
) -> dict[str, np.ndarray]:
    """Map the selected exterior view and apply 224-square resize-with-pad."""
    from openpi_client.image_tools import resize_with_pad

    base = _tensor_to_hwc_uint8(sample[exterior_camera_key])
    wrist = _tensor_to_hwc_uint8(sample["wrist_image_left"])
    if base.shape != wrist.shape:
        raise ValueError(
            "The official DROID mapping requires exterior and wrist images to "
            f"share a shape, got {base.shape} and {wrist.shape}."
        )
    raw_images = {
        "base_0_rgb": base,
        "left_wrist_0_rgb": wrist,
        "right_wrist_0_rgb": np.zeros_like(base),
    }
    return {key: resize_with_pad(image, 224, 224) for key, image in raw_images.items()}


def _write_episode_video(
    dataset: Any,
    output_path: Path,
    fps: float,
    exterior_camera_key: str,
    imageio_ffmpeg: Any,
) -> None:
    """Write a VS Code-compatible H.264, three-panel pi05_droid input video."""
    frame_size = (224 * len(_MODEL_IMAGE_KEYS), 224)
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        frame_size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    writer.send(None)

    try:
        for frame_index in range(len(dataset)):
            images = _get_pi05_droid_images(dataset[frame_index], exterior_camera_key)
            frame = np.ascontiguousarray(
                np.concatenate([images[key] for key in _MODEL_IMAGE_KEYS], axis=1)
            )
            writer.send(frame)
            if (frame_index + 1) % 50 == 0 or frame_index + 1 == len(dataset):
                print(f"Encoded {frame_index + 1}/{len(dataset)} frames", flush=True)
    finally:
        writer.close()


def main() -> None:
    args = _build_parser().parse_args()
    if args.episode_index < 0:
        raise ValueError("--episode-index must be non-negative.")
    if not args.dataset_path.joinpath("meta", "info.json").is_file():
        raise FileNotFoundError(
            f"No LeRobot dataset metadata found under {args.dataset_path}."
        )

    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Per-frame embedded-PNG decoding is substantially slower when every small
    # image operation fan-outs to the host's full PyTorch thread pool.
    torch.set_num_threads(1)

    dataset = LeRobotDataset(
        args.dataset_path.name,
        root=args.dataset_path,
        episodes=[args.episode_index],
        download_videos=False,
    )
    fps = args.fps if args.fps is not None else float(dataset.fps)
    if fps <= 0:
        raise ValueError(f"Output FPS must be positive, got {fps}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"episode_{args.episode_index:06d}_pi05_droid_inputs.mp4"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

    exterior_camera_key = _EXTERIOR_CAMERA_KEYS[args.external_camera]
    _write_episode_video(
        dataset,
        output_path,
        fps,
        exterior_camera_key,
        _require_imageio_ffmpeg(),
    )
    print(f"Wrote {len(dataset)} frames at {fps:g} FPS to {output_path}")
    print(
        f"Panel order: base_0_rgb ({exterior_camera_key}), "
        "left_wrist_0_rgb (wrist_image_left), right_wrist_0_rgb (masked zeros)."
    )


if __name__ == "__main__":
    main()
