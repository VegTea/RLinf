# Copyright 2025 The RLinf Authors.
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

"""Validation and normalization helpers for IsaacLab evaluation scenarios."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


VECTOR_LENGTHS = {
    "cube_1_pos": 3,
    "cube_1_rpy": 3,
    "cube_2_pos": 3,
    "cube_2_rpy": 3,
    "cube_3_pos": 3,
    "cube_3_rpy": 3,
    "table_cam_pos": 3,
    "table_cam_rot": 4,
}
REQUIRED_KEYS = ("id", "table_asset", *VECTOR_LENGTHS)


def validate_scenario_record(record: Mapping, *, location: str = "record") -> None:
    """Validate fields needed to reload a captured evaluation scenario."""
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"{location} is missing required fields: {missing}")
    if not str(record["id"]).strip():
        raise ValueError(f"{location}.id must be non-empty")
    if not str(record["table_asset"]).strip():
        raise ValueError(f"{location}.table_asset must be non-empty")
    for key, expected_length in VECTOR_LENGTHS.items():
        value = record[key]
        if not isinstance(value, (list, tuple)) or len(value) != expected_length:
            raise ValueError(
                f"{location}.{key} must be a {expected_length}-element array"
            )
        try:
            [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location}.{key} must contain numbers") from exc


def validate_scenario_file(path: str | Path) -> int:
    """Validate a JSONL scenario file and return its record count."""
    scenario_path = Path(path).expanduser().resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario file does not exist: {scenario_path}")

    seen_ids: set[str] = set()
    count = 0
    with scenario_path.open(encoding="utf-8") as scenario_fp:
        for line_number, line in enumerate(scenario_fp, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{scenario_path}:{line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{scenario_path}:{line_number} must contain a JSON object"
                )
            validate_scenario_record(
                record, location=f"{scenario_path}:{line_number}"
            )
            scenario_id = str(record["id"])
            if scenario_id in seen_ids:
                raise ValueError(
                    f"{scenario_path}:{line_number} has duplicate id {scenario_id!r}"
                )
            seen_ids.add(scenario_id)
            count += 1
    if count == 0:
        raise ValueError(f"Scenario file contains no records: {scenario_path}")
    return count


def normalize_eval_scenarios(records: Iterable[Mapping]) -> list[dict]:
    """Assign loader-safe IDs while preserving source scenario provenance."""
    normalized = []
    for index, original in enumerate(records):
        record = dict(original)
        source_scenario_id = record.pop("_source_scenario_id", None)
        if source_scenario_id is None:
            source_scenario_id = record.get("source_scenario_id")
        record["id"] = f"eval_{index:06d}"
        if source_scenario_id is not None:
            record["source_scenario_id"] = str(source_scenario_id)
        validate_scenario_record(record, location=f"eval scenario {index}")
        normalized.append(record)
    return normalized


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="SCENARIO_JSONL", required=True)
    args = parser.parse_args()
    count = validate_scenario_file(args.validate)
    print(f"Validated {count} scenario records: {Path(args.validate).resolve()}")


if __name__ == "__main__":
    _main()
