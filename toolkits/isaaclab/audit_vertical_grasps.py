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

"""CPU-only bounded search for vertical grasp paths; not a collision certificate."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from toolkits.isaaclab.privileged_stack_controller import PandaIK

Q0 = np.array([0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952])


def vertical_path(ik, position, yaw, seed):
    """Check descending IK samples and reject discontinuous joint branches."""
    rotation = (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("x", 180, degrees=True).as_matrix()
    )
    q = seed.copy()
    path = []
    for z in np.linspace(position[2] + 0.132, position[2] + 0.002, 14):
        target = np.array([position[0], position[1], z])
        try:
            solution = ik.solve(target, rotation, q)
        except RuntimeError:
            return {
                "ok": False,
                "failed_tcp_z": float(z),
                "reason": "IK residual above tolerance",
            }
        if path and np.max(np.abs(solution - q)) > 0.25:
            return {
                "ok": False,
                "failed_tcp_z": float(z),
                "reason": "joint jump >0.25 rad per 1cm waypoint",
            }
        path.append(solution)
        q = solution
    path = np.array(path)
    return {
        "ok": True,
        "yaw_rad": float(yaw),
        "min_hard_limit_margin_rad": float(
            np.minimum(path - ik.lo, ik.hi - path).min() + 0.06
        ),
        "path_q_rad": path.tolist(),
    }


def check_target(job):
    """Try the default branch, then a bounded deterministic multistart search."""
    usd, key, position, cube_yaw, extra_starts = job
    ik = PandaIK(usd)
    baseline = vertical_path(ik, position, 0.0, Q0)
    result = {
        "key": key,
        "position": position,
        "cube_yaw_rad": cube_yaw,
        "default_yaw0": baseline,
        "search_attempts": 1,
    }
    if baseline["ok"]:
        result.update(status="default_path_found", witness=baseline)
        return result
    rng = np.random.default_rng(42)
    seeds = [Q0] + [
        rng.uniform(ik.lo + 0.001, ik.hi - 0.001) for _ in range(extra_starts)
    ]
    # Includes cube-face-aligned grasps and canonical vertical tool yaws.
    yaws = sorted(
        {
            round(float(y), 8)
            for y in [0, np.pi / 2, np.pi, -np.pi / 2]
            + [cube_yaw + k * np.pi / 2 for k in range(4)]
        }
    )
    for seed in seeds:
        for yaw in yaws:
            if np.array_equal(seed, Q0) and yaw == 0:
                continue
            found = vertical_path(ik, position, yaw, seed)
            result["search_attempts"] += 1
            if found["ok"]:
                result.update(status="alternative_path_found", witness=found)
                return result
    result.update(status="no_path_found_in_bounded_search", witness=None)
    return result


def main():
    """Write per-scenario/cube CSV and reproducible IK witnesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--extra-starts", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        ik = PandaIK(args.usd)
        good = vertical_path(ik, [0.5, 0, 0.0203], 0, Q0)
        assert good["ok"], good
        assert good["min_hard_limit_margin_rad"] >= 0.06 - 1e-8
        bad = vertical_path(ik, [3, 0, 0.0203], 0, Q0)
        assert not bad["ok"], bad
        print("Vertical path self-check passed")
        return
    records = [
        json.loads(line)
        for line in args.scenarios.read_text().splitlines()
        if line.strip()
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {}
    references = []
    for record in records:
        for cube in range(1, 4):
            position = record[f"cube_{cube}_pos"]
            rpy = record[f"cube_{cube}_rpy"]
            if abs(rpy[0]) + abs(rpy[1]) > 1e-6:
                raise ValueError("This audit assumes initially upright cubes")
            key = json.dumps([position, rpy[2]])
            targets[key] = (args.usd, key, position, rpy[2], args.extra_starts)
            references.append(
                {
                    "scenario_id": record["id"],
                    "cube": cube,
                    "color": ["blue", "red", "green"][cube - 1],
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                    "key": key,
                }
            )
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with (args.output / "progress.jsonl").open("x", buffering=1) as progress:
            for result in pool.map(check_target, targets.values()):
                results[result["key"]] = result
                progress.write(json.dumps(result) + "\n")
                print(
                    len(results),
                    len(targets),
                    result["position"],
                    result["status"],
                    flush=True,
                )
    for row in references:
        r = results[row.pop("key")]
        row.update(
            status=r["status"],
            default_failure_height=r["default_yaw0"].get("failed_tcp_z"),
            search_attempts=r["search_attempts"],
        )
    with (args.output / "per_scenario_cube.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)
    summary = {
        "scenarios": len(records),
        "unique_targets": len(targets),
        "cube_instances": len(references),
        "extra_starts": args.extra_starts,
        "collision_checked": False,
        "criteria": "TCP z=cube z+0.132 down to z+0.002; 1cm samples; vertical orientation; hard-limit margin 0.06rad; IK error<=3mm/0.03rad; inter-sample joint jump<=0.25rad. Reverse path is kinematically identical; inter-sample collisions/initial approach not checked.",
        "unique_target_counts": {
            s: sum(r["status"] == s for r in results.values())
            for s in [
                "default_path_found",
                "alternative_path_found",
                "no_path_found_in_bounded_search",
            ]
        },
        "affected_scenarios": {
            s: sorted({r["scenario_id"] for r in references if r["status"] == s})
            for s in ["alternative_path_found", "no_path_found_in_bounded_search"]
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
