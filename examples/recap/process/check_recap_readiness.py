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

"""Preflight checks for running RECAP on OpenPI/YAM data.

This script is intentionally read-only by default. It validates an OpenPI JAX
checkpoint layout, checks LeRobot dataset schemas, and reports whether the data
can use the dual-arm YAM RECAP adapter.

Examples:
    python examples/recap/process/check_recap_readiness.py \
        --checkpoint-dir /path/to/openpi_jax_ckpt \
        --expert-data /path/to/expert_lerobot \
        --rollout-data /path/to/rollout_lerobot \
        --hitl-data /path/to/hitl_lerobot

    python examples/recap/process/check_recap_readiness.py \
        --checkpoint-dir /path/to/openpi_jax_ckpt \
        --jax-inspect \
        --converted-output checkpoints/torch/yam_pi05_sft
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERTER = REPO_ROOT / "rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py"

np = None
pq = None

REQUIRED_COLUMNS = {"episode_index", "frame_index"}
ACTION_COLUMNS = ("action", "actions")
STATE_COLUMNS = (
    "observation.state",
    "state",
)
PROMPT_COLUMNS = ("task", "task_index", "prompt")

YAM_IMAGE_COLUMNS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
YAM_STATE_COLUMNS = (
    "observation.state",
)
YAM_ACTION_COLUMNS = ("action",)


@dataclass
class DatasetSpec:
    path: Path
    kind: str
    robot_type: str = "yam"


def _load_array_deps() -> None:
    """Import dataset inspection dependencies lazily."""
    global np, pq
    if np is None:
        import numpy as _np

        np = _np
    if pq is None:
        import pyarrow.parquet as _pq

        pq = _pq


def _status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _warn(ok: bool) -> str:
    return "OK" if ok else "WARN"


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


def _shape_of_value(value: Any) -> tuple[int, ...] | str:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        return "struct"
    arr = np.asarray(value)
    return tuple(arr.shape)


def _first_value_shape(parquet_file: Path, column: str) -> tuple[int, ...] | str | None:
    try:
        table = pq.read_table(parquet_file, columns=[column])
    except Exception:
        return None
    if table.num_rows == 0:
        return None
    return _shape_of_value(table.column(column)[0])


def _find_parquets(dataset_path: Path, max_files: int) -> list[Path]:
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*.parquet"))[:max_files]


def _read_tasks(dataset_path: Path) -> dict[int, str]:
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = {}
    if not tasks_path.exists():
        return tasks
    with tasks_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_index = int(item.get("task_index", len(tasks)))
            tasks[task_index] = str(item.get("task", ""))
    return tasks


def _read_info_features(dataset_path: Path) -> set[str]:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return set()
    with info_path.open() as f:
        info = json.load(f)
    return set(info.get("features", {}))


def _compute_return_preview(
    parquet_file: Path,
    dataset_kind: str,
    gamma: float,
    failure_reward: float,
) -> tuple[dict[str, float], list[str]]:
    warnings = []
    cols = ["episode_index", "frame_index", "is_success", "reward"]
    available_cols = set(pq.ParquetFile(parquet_file).schema_arrow.names)
    table = pq.read_table(
        parquet_file, columns=[c for c in cols if c in available_cols]
    )
    names = set(table.column_names)
    if "episode_index" not in names:
        return {}, ["missing episode_index; cannot preview returns"]
    if dataset_kind == "reward" and "reward" not in names:
        return {}, ["reward data needs reward to preview returns"]
    if dataset_kind not in {"sft", "reward", "failure"} and "is_success" not in names:
        return {}, ["rollout/HITL data needs is_success to preview returns"]

    ep_indices = table.column("episode_index").to_numpy().astype(np.int64)
    success_values = (
        table.column("is_success").to_pylist() if "is_success" in names else None
    )
    rewards = []
    returns = []

    starts = np.concatenate([[0], np.where(np.diff(ep_indices) != 0)[0] + 1])
    ends = np.concatenate([starts[1:], [len(ep_indices)]])
    for start, end in zip(starts, ends, strict=True):
        length = end - start
        if length <= 0:
            continue
        if dataset_kind == "reward":
            ep_rewards = (
                table.column("reward")
                .to_numpy()
                .astype(np.float32, copy=False)[start:end]
            )
            ep_returns = np.zeros(length, dtype=np.float32)
            ep_returns[-1] = ep_rewards[-1]
            for i in range(length - 2, -1, -1):
                ep_returns[i] = ep_rewards[i] + gamma * ep_returns[i + 1]
        else:
            is_success = dataset_kind == "sft"
            if dataset_kind not in {"sft", "failure"} and success_values is not None:
                is_success = bool(success_values[end - 1])
            ep_rewards = np.full(length, -1.0, dtype=np.float32)
            ep_rewards[-1] = 0.0 if is_success else failure_reward
            ep_returns = np.zeros(length, dtype=np.float32)
            ep_returns[-1] = ep_rewards[-1]
            for i in range(length - 2, -1, -1):
                ep_returns[i] = ep_rewards[i] + gamma * ep_returns[i + 1]
        rewards.extend(ep_rewards.tolist())
        returns.extend(ep_returns.tolist())

    if not returns:
        return {}, ["no rows found for return preview"]
    ret = np.asarray(returns, dtype=np.float32)
    rew = np.asarray(rewards, dtype=np.float32)
    return {
        "return_min": float(ret.min()),
        "return_max": float(ret.max()),
        "return_mean": float(ret.mean()),
        "reward_min": float(rew.min()),
        "reward_max": float(rew.max()),
    }, warnings


def check_checkpoint(
    checkpoint_dir: Path | None,
    config_name: str,
    jax_inspect: bool,
    converted_output: Path | None,
    load_converted: bool,
) -> bool:
    if checkpoint_dir is None:
        return True

    _print_header("Checkpoint")
    ok = True
    print(f"path: {checkpoint_dir}")
    exists = checkpoint_dir.exists()
    print(f"[{_status(exists)}] checkpoint directory exists")
    if not exists:
        return False

    children = {p.name for p in checkpoint_dir.iterdir()}
    orbax_markers = {
        "_CHECKPOINT_METADATA",
        "_METADATA",
        "checkpoint",
        "params",
        "ocdbt.process_0",
    }
    looks_jax = bool(children & orbax_markers) or any(
        p.name.startswith("step") for p in checkpoint_dir.iterdir()
    )
    print(f"[{_warn(looks_jax)}] looks like an Orbax/JAX checkpoint")

    assets_candidates = [
        checkpoint_dir / "assets",
        checkpoint_dir.parent / "assets",
        checkpoint_dir / "norm_stats.json",
        checkpoint_dir / "norm_stats",
    ]
    has_assets = any(p.exists() for p in assets_candidates)
    print(f"[{_warn(has_assets)}] assets or norm stats found near checkpoint")
    if not has_assets:
        print("  expected one of:", ", ".join(str(p) for p in assets_candidates))

    if jax_inspect:
        cmd = [
            sys.executable,
            str(CONVERTER),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--config-name",
            config_name,
            "--inspect-only",
        ]
        print("running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        ok = ok and result.returncode == 0
        print(f"[{_status(result.returncode == 0)}] JAX inspect")

    if converted_output is not None:
        cmd = [
            sys.executable,
            str(CONVERTER),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--config-name",
            config_name,
            "--output-path",
            str(converted_output),
        ]
        print("running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        ok = ok and result.returncode == 0
        print(f"[{_status(result.returncode == 0)}] JAX -> PyTorch conversion")
        ok = check_converted_checkpoint(converted_output, load_converted) and ok

    return ok


def check_converted_checkpoint(converted_dir: Path, load_model: bool) -> bool:
    _print_header("Converted PyTorch Checkpoint")
    ok = True
    model_file = converted_dir / "model.safetensors"
    assets_dir = converted_dir / "assets"
    norm_stats_file = converted_dir / "norm_stats.json"
    print(f"path: {converted_dir}")
    print(f"[{_status(model_file.exists())}] model.safetensors")
    print(f"[{_warn(assets_dir.exists() or norm_stats_file.exists())}] assets/norm stats")
    ok = ok and model_file.exists()

    if load_model:
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.openpi_cfg import get_model

        cfg = OmegaConf.create(
            {
                "model_path": str(converted_dir),
                "openpi": {
                    "config_name": "pi05_yam",
                    "train_expert_only": False,
                    "cfgrl_guidance_scale": 1.0,
                    "unconditional_prob": 0.1,
                    "guidance_type": "positive",
                    "positive_only_conditional": True,
                },
            }
        )
        try:
            get_model(cfg)
        except Exception as exc:
            ok = False
            print(f"[FAIL] RLinf CFG model load: {exc}")
        else:
            print("[OK] RLinf CFG model load")

    return ok


def check_dataset(
    spec: DatasetSpec, max_files: int, gamma: float, failure_reward: float
) -> bool:
    _load_array_deps()
    _print_header(f"Dataset: {spec.kind} ({spec.robot_type})")
    ok = True
    dataset_path = spec.path
    print(f"path: {dataset_path}")
    exists = dataset_path.exists()
    print(f"[{_status(exists)}] dataset path exists")
    if not exists:
        return False

    for subdir in ("data", "meta"):
        found = (dataset_path / subdir).exists()
        print(f"[{_status(found)}] {subdir}/ directory")
        ok = ok and found

    parquets = _find_parquets(dataset_path, max_files)
    print(f"[{_status(bool(parquets))}] parquet files found: {len(parquets)} checked")
    if not parquets:
        return False

    schema_names: set[str] = set()
    for parquet_file in parquets:
        schema_names.update(pq.ParquetFile(parquet_file).schema_arrow.names)
    info_features = _read_info_features(dataset_path)
    dataset_features = schema_names | info_features

    missing_required = REQUIRED_COLUMNS - schema_names
    print(f"[{_status(not missing_required)}] required index columns")
    if missing_required:
        print(f"  missing: {sorted(missing_required)}")
        ok = False

    action_col = next((c for c in ACTION_COLUMNS if c in dataset_features), None)
    state_col = next((c for c in STATE_COLUMNS if c in dataset_features), None)
    image_cols = sorted(c for c in dataset_features if "image" in c.lower())
    prompt_col = next((c for c in PROMPT_COLUMNS if c in dataset_features), None)

    print(f"[{_status(action_col is not None)}] action column: {action_col}")
    print(f"[{_status(state_col is not None)}] state column: {state_col}")
    print(f"[{_status(bool(image_cols))}] image columns: {image_cols[:8]}")
    print(f"[{_warn(prompt_col is not None)}] prompt/task column: {prompt_col}")
    ok = ok and action_col is not None and state_col is not None and bool(image_cols)

    needs_success = spec.kind in {"rollout", "hitl"}
    has_success = "is_success" in schema_names
    print(f"[{_status((not needs_success) or has_success)}] is_success for {spec.kind}")
    ok = ok and ((not needs_success) or has_success)

    sample_file = parquets[0]
    if action_col:
        print(f"  action shape sample: {_first_value_shape(sample_file, action_col)}")
    if state_col:
        print(f"  state shape sample: {_first_value_shape(sample_file, state_col)}")
    for image_col in image_cols[:3]:
        shape = _first_value_shape(sample_file, image_col)
        print(f"  image shape sample ({image_col}): {shape}")

    tasks = _read_tasks(dataset_path)
    print(f"[{_warn(bool(tasks) or prompt_col is not None)}] task descriptions available")

    preview, warnings = _compute_return_preview(
        sample_file, spec.kind, gamma=gamma, failure_reward=failure_reward
    )
    if warnings:
        print(f"[WARN] return preview: {'; '.join(warnings)}")
    else:
        finite = all(math.isfinite(v) for v in preview.values())
        print(f"[{_status(finite)}] return dry-run preview: {preview}")
        ok = ok and finite

    if spec.robot_type == "yam":
        yam_ok = bool(set(YAM_IMAGE_COLUMNS) & dataset_features)
        yam_ok = yam_ok and bool(set(YAM_STATE_COLUMNS) & dataset_features)
        yam_ok = yam_ok and bool(set(YAM_ACTION_COLUMNS) & dataset_features)
        print(f"[{_warn(yam_ok)}] YAM adapter compatibility")
        if not yam_ok:
            print("  expected YAM image columns:", list(YAM_IMAGE_COLUMNS))
            print("  expected YAM state column:", list(YAM_STATE_COLUMNS))
            print("  expected YAM action column:", list(YAM_ACTION_COLUMNS))
            if "actions" in schema_names:
                print(
                    "  found legacy 'actions'; either rename it to 'action' for "
                    "the YAM templates or set a custom OpenPI data config."
                )
            print("  If your YAM schema uses different names, add them to:")
            print("    examples/recap/process/compute_advantages.py KEY_MAPPINGS['yam']")
            print("    rlinf/data/datasets/recap/value_model.py _REPACK_KEYS['yam']")
        else:
            print("  expected state/action effective dim: 14")
            print("  expected cameras: cam_high, cam_left_wrist, cam_right_wrist")

    return ok


def parse_dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs = []
    for path in args.expert_data:
        specs.append(DatasetSpec(Path(path), "sft", args.robot_type))
    for path in args.rollout_data:
        specs.append(DatasetSpec(Path(path), "rollout", args.robot_type))
    for path in args.hitl_data:
        specs.append(DatasetSpec(Path(path), "hitl", args.robot_type))
    for raw in args.dataset:
        parts = raw.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                "--dataset must use PATH:TYPE or PATH:TYPE:ROBOT_TYPE, "
                f"got {raw!r}"
            )
        path, kind = parts[0], parts[1]
        robot_type = parts[2] if len(parts) == 3 else args.robot_type
        specs.append(DatasetSpec(Path(path), kind, robot_type))
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--converter-config-name",
        default="pi05_yam",
        help="OpenPI config name used by the JAX->PyTorch converter.",
    )
    parser.add_argument("--jax-inspect", action="store_true")
    parser.add_argument("--converted-output", type=Path)
    parser.add_argument("--check-converted-dir", type=Path)
    parser.add_argument("--load-converted-model", action="store_true")
    parser.add_argument("--expert-data", action="append", default=[])
    parser.add_argument("--rollout-data", action="append", default=[])
    parser.add_argument("--hitl-data", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--robot-type", default="yam")
    parser.add_argument("--max-parquet-files", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--failure-reward", type=float, default=-300.0)
    args = parser.parse_args()

    all_ok = True
    all_ok = (
        check_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            config_name=args.converter_config_name,
            jax_inspect=args.jax_inspect,
            converted_output=args.converted_output,
            load_converted=args.load_converted_model,
        )
        and all_ok
    )

    if args.check_converted_dir is not None:
        all_ok = (
            check_converted_checkpoint(
                args.check_converted_dir, load_model=args.load_converted_model
            )
            and all_ok
        )

    for spec in parse_dataset_specs(args):
        all_ok = (
            check_dataset(
                spec,
                max_files=args.max_parquet_files,
                gamma=args.gamma,
                failure_reward=args.failure_reward,
            )
            and all_ok
        )

    _print_header("Summary")
    if all_ok:
        print("[OK] RECAP preflight checks passed.")
        return 0
    print("[FAIL] RECAP preflight found blocking issues. See messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
