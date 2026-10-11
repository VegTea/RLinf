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

"""Stream batch IK diagnostics into per-scenario joint-limit evidence."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def limit_masks(q, target, limits):
    """Separate proximity, physical contact and same-direction blocked commands."""
    lo, hi = limits[:, 0], limits[:, 1]
    near = (q - lo <= 0.01) | (hi - q <= 0.01)
    contact = (q - lo <= 0.001) | (hi - q <= 0.001)
    outside = (target < lo - 0.001) | (target > hi + 0.001)
    blocked = ((q - lo <= 0.001) & (target < lo - 0.001)) | (
        (hi - q <= 0.001) & (target > hi + 0.001)
    )
    return near, contact, outside, blocked


def main():
    """Require complete original-scenario coverage before reporting results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        limits = np.array([[-1.0, 1.0]] * 3)
        near, hit, out, blocked = limit_masks(
            np.array([-1.0, 0.0, 1.0]), np.array([-1.1, 2.0, 0.8]), limits
        )
        assert hit.tolist() == [True, False, True]
        assert out.tolist() == [True, True, False]
        assert blocked.tolist() == [True, False, False]
        assert near.tolist() == [True, False, True]
        print("Joint-limit classification self-check passed")
        return
    root = args.run
    if (root / "exit_code").read_text().strip() != "0":
        raise RuntimeError("Evaluation did not exit successfully")
    expected = {
        json.loads(line)["id"]
        for line in args.scenarios.read_text().splitlines()
        if line.strip()
    }
    episodes = [
        json.loads(p.read_text()) for p in (root / "per_env_results").glob("*.json")
    ]
    ids = [str(r["scenario_id"]) for r in episodes]
    assert len(ids) == len(expected) and set(ids) == expected, (
        "Missing/duplicate scenario episodes"
    )
    episode_map = {(r["worker_rank"], r["env_id"]): r for r in episodes}
    assert len(episode_map) == len(episodes), "Multiple episodes in one slot"
    aggregates = {}
    for path in sorted(root.glob("ik_trace.jsonl.rank*")):
        rank = int(path.name.split(".rank")[-1])
        with path.open() as stream:
            meta = json.loads(next(stream))
            dt, decimation = meta["physics_dt"], meta["decimation"]
            joint_indexes = [
                meta["joint_names"].index(n) for n in meta["arm_joint_names"]
            ]
            for line in stream:
                r = json.loads(line)
                key = (rank, r["env_id"])
                episode = episode_map[key]
                assert str(r["scenario_id"]) == str(episode["scenario_id"]), (
                    "Scenario changed within rollout"
                )
                if key not in aggregates:
                    aggregates[key] = {
                        "count": 0,
                        "posts": 0,
                        "near": np.zeros(7, int),
                        "contact": np.zeros(7, int),
                        "outside": np.zeros(7, int),
                        "run": np.zeros(7, int),
                        "maxrun": np.zeros(7, int),
                        "error_steps": 0,
                        "blocked_error_steps": 0,
                        "down_up_steps": 0,
                        "first_blocked_s": None,
                        "last_index": -1,
                        "head": None,
                        "prev_q": None,
                    }
                s = aggregates[key]
                if r["event"] == "ik":
                    index = (r["step"] - 1) * decimation + r["substep"]
                    assert index == s["last_index"] + 1, (
                        "Missing/out-of-order physical sample"
                    )
                    s["last_index"] = index
                    limits = np.array(meta["joint_pos_limits_per_env"][r["env_id"]])[
                        joint_indexes
                    ]
                    q = np.array(r["arm_joint_pos"])
                    target = np.array(r["ik_joint_target"])
                    near, hit, out, blocked = limit_masks(q, target, limits)
                    # Do not bridge a discontinuous internal reset in sustained counts.
                    if (
                        s["prev_q"] is not None
                        and np.max(np.abs(q - s["prev_q"])) > 0.5
                    ):
                        s["run"][:] = 0
                    s["prev_q"] = q
                    s["run"] = np.where(blocked, s["run"] + 1, 0)
                    s["maxrun"] = np.maximum(s["maxrun"], s["run"])
                    s["near"] += near
                    s["contact"] += hit
                    s["outside"] += out
                    s["count"] += 1
                    if blocked.any() and s["first_blocked_s"] is None:
                        s["first_blocked_s"] = index * dt
                    if r["substep"] == 0:
                        s["head"] = r
                        s["step_blocked"] = False
                    s["step_blocked"] |= bool(blocked.any())
                elif r["event"] == "post_step":
                    s["posts"] += 1
                    head = s["head"]
                    assert head["step"] == r["step"]
                    # Exclude a reset jump when comparing command with the post-step state.
                    endq = np.array(r["joint_pos_all"])[joint_indexes]
                    if np.max(np.abs(endq - s["prev_q"])) < 0.5:
                        error = np.linalg.norm(
                            np.array(head["target_tcp_pos"]) - r["tcp_pos"]
                        )
                        s["error_steps"] += int(error > 0.01)
                        s["blocked_error_steps"] += int(
                            error > 0.01 and s["step_blocked"]
                        )
                        s["down_up_steps"] += int(
                            s["step_blocked"]
                            and head["processed_arm_action"][2] < 0
                            and r["tcp_pos"][2] > head["tcp_pos"][2] + 0.0001
                        )
            for key, s in aggregates.items():
                if key[0] != rank:
                    continue
                s["dt"] = dt
                s["decimation"] = decimation
                s["joint_names"] = meta["arm_joint_names"]
    assert set(aggregates) == set(episode_map), "Diagnostic coverage incomplete"
    rows = []
    for key, s in sorted(aggregates.items()):
        assert s["count"] == 450 * s["decimation"] and s["posts"] == 450, (
            "Expected 450 complete control steps"
        )
        episode = episode_map[key]
        video = Path(episode["video_path"])
        assert video.is_file() and video.stat().st_size > 0, f"Missing video {video}"

        def names(mask):
            return ";".join(n for n, b in zip(s["joint_names"], mask) if b)

        duration = s["maxrun"] * s["dt"]
        sustained = duration >= 0.5
        rows.append(
            {
                "scenario_id": episode["scenario_id"],
                "worker_rank": key[0],
                "env_id": key[1],
                "success": episode["policy_success"],
                "near_limit_joints": names(s["near"] > 0),
                "contact_joints": names(s["contact"] > 0),
                "ik_outside_joints": names(s["outside"] > 0),
                "sustained_blocked_joints": names(sustained),
                "max_blocked_duration_s": float(duration.max()),
                "first_blocked_s": s["first_blocked_s"],
                "tracking_error_steps": s["error_steps"],
                "blocked_tracking_error_steps": s["blocked_error_steps"],
                "blocked_command_down_actual_up_steps": s["down_up_steps"],
                "video_path": str(video.relative_to(root.resolve()))
                if video.is_absolute()
                else str(video),
            }
        )
    with (root / "joint_limit_report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    groups = {
        name: [r["scenario_id"] for r in rows if pred(r)]
        for name, pred in {
            "success": lambda r: r["success"],
            "failure": lambda r: not r["success"],
            "any_contact": lambda r: bool(r["contact_joints"]),
            "any_ik_outside": lambda r: bool(r["ik_outside_joints"]),
            "sustained_blocked": lambda r: bool(r["sustained_blocked_joints"]),
            "failed_and_sustained_blocked": lambda r: (
                not r["success"] and bool(r["sustained_blocked_joints"])
            ),
            "successful_and_sustained_blocked": lambda r: (
                r["success"] and bool(r["sustained_blocked_joints"])
            ),
        }.items()
    }
    summary = {
        "total": len(rows),
        "counts": {k: len(v) for k, v in groups.items()},
        "scenario_ids": groups,
        "definitions": {
            "near_rad": 0.01,
            "contact_rad": 0.001,
            "ik_beyond_rad": 0.001,
            "sustained_same_direction_blocked_seconds": 0.5,
            "position_error_m": 0.01,
        },
        "caveat": "Observed limit events, not proof that limits caused every failure. Only seven arm joints; normal finger end stops excluded.",
    }
    (root / "joint_limit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
