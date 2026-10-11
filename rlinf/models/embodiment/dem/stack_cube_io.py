# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stack-Cube DEM protocol, independent of the DEM/IsaacLab installations."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

INSTRUCTION = "Stack the red block on the blue block, then stack the green block on the red block."
DATASET_REVISION = "2e27f04b2e31f293fd020164917e4cce09e108d6"
DEM_REVISION = "e5c8f6a81dca59bdfea2d1de067a7ec6f8a3e0ea"


STATE_ROTATION_ENCODINGS = ("native_xyzw", "legacy_wxyz_as_xyzw")


def environment_state_to_dataset(
    states: torch.Tensor,
    native_quat_xyzw: torch.Tensor,
    *,
    encoding: str,
) -> torch.Tensor:
    """Build stored-format Euler features from native IsaacLab 3 quaternions.

    Ignore the legacy pi05 rotation-vector columns; preserve position and fingers.
    The manifest must explicitly declare the dataset encoding. The legacy option
    reproduces WXYZ values interpreted by SciPy as XYZW; it is an acquisition
    hypothesis, NOT an implicit correction or evidence of a verified contract.
    """
    if states.ndim != 2 or states.shape[1] != 8 or not torch.isfinite(states).all():
        raise ValueError("Expected finite Stack-Cube states [B, 8]")
    if encoding not in STATE_ROTATION_ENCODINGS:
        raise ValueError(f"Unknown state rotation encoding: {encoding}")
    if native_quat_xyzw.shape != (states.shape[0], 4):
        raise ValueError("Expected native XYZW quaternions [B, 4]")
    quat = native_quat_xyzw.detach().float().cpu().numpy()
    if not np.isfinite(quat).all() or np.any(
        np.abs(np.linalg.norm(quat, axis=1) - 1) > 1e-3
    ):
        raise ValueError("Native XYZW quaternions must be finite and unit length")
    if encoding == "legacy_wxyz_as_xyzw":
        quat = quat[:, [3, 0, 1, 2]]
    result = states.detach().float().clone()
    result[:, 3:6] = torch.as_tensor(
        Rotation.from_quat(quat).as_euler("xyz"),
        dtype=result.dtype,
        device=result.device,
    )
    return result


def prepare_stack_actions(actions: torch.Tensor, execute_steps: int) -> torch.Tensor:
    """Select raw controller commands, without another normalization/IK scaling."""
    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise ValueError("Expected actions [B, H, 7]")
    if not 0 < execute_steps <= actions.shape[1]:
        raise ValueError("Execution length must be between 1 and prediction horizon")
    if not torch.isfinite(actions).all():
        raise ValueError("Policy produced non-finite actions")
    result = actions[:, :execute_steps].float().clone()
    # IsaacLab's BinaryJointPositionAction closes only for values < 0.
    result[..., -1] = torch.where(result[..., -1] < 0, -1.0, 1.0)
    return result.contiguous()


def rotation_audit(states: np.ndarray) -> dict:
    """Compare consecutive orientation changes under two hypotheses, not a proof."""
    result = {}
    for name, rot in (
        ("euler_xyz", Rotation.from_euler("xyz", states[:, 3:6])),
        ("rotation_vector", Rotation.from_rotvec(states[:, 3:6])),
    ):
        changes = (rot[:-1].inv() * rot[1:]).magnitude()
        result[name] = {"max_step_radians": float(changes.max(initial=0))}
    return result
