# Copyright 2025 The RLinf Authors.
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

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCENARIO_ASSET_ROOT = Path(
    os.environ.get(
        "RLINF_SCENARIO_ASSET_ROOT",
        Path(__file__).resolve().parents[2] / "assets_isaaclab",
    )
)
DEFAULT_TABLE_ASSET_ROOT = (
    DEFAULT_SCENARIO_ASSET_ROOT / "SeattleLabTable" / "color_tables"
)


def _first_existing(paths):
    for path in paths:
        if path and Path(path).exists():
            return path
    return None


def build_table_texture_registry():
    # 旧运行时table材质替换逻辑。
    # 该 registry 保留用于兼容/回退，不再作为主路径使用。
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

    candidates = {
        "default": [
            f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png",
        ],
        "steel_stainless": [
            f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
        ],
        "brass": [
            f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
        ],
        "copper": [
            f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
        ],
        "aluminum_cast": [
            f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png",
        ],
        "aluminum_anodized": [
            f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized/Aluminum_Anodized_BaseColor.png",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized/Aluminum_Anodized_BaseColor.png",
        ],
        "brushed_antique_copper": [
            f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Metals/Brushed_Antique_Copper/Brushed_Antique_Copper_BaseColor.png",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brushed_Antique_Copper/Brushed_Antique_Copper_BaseColor.png",
        ],
    }

    registry = {}
    for key, path_candidates in candidates.items():
        resolved = _first_existing(path_candidates)
        if resolved is not None:
            registry[key] = resolved
    return registry


def _candidate_table_asset_paths(
    table_asset: str,
    table_asset_root: str | Path | None = None,
) -> list[Path]:
    asset_root = (
        Path(table_asset_root)
        if table_asset_root is not None
        else DEFAULT_TABLE_ASSET_ROOT
    )
    requested_path = Path(table_asset)

    candidates: list[Path] = []
    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.append(asset_root / requested_path)

    if requested_path.suffix == ".usd":
        alternate = requested_path.with_suffix(".usda")
    elif requested_path.suffix == ".usda":
        alternate = requested_path.with_suffix(".usd")
    else:
        alternate = None
        candidates.append(asset_root / f"{requested_path.name}.usd")
        candidates.append(asset_root / f"{requested_path.name}.usda")

    if alternate is not None:
        candidates.append(
            alternate if alternate.is_absolute() else asset_root / alternate
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_table_asset_path(
    table_asset: str,
    table_asset_root: str | Path | None = None,
    must_exist: bool = False,
) -> str:
    candidates = _candidate_table_asset_paths(
        table_asset, table_asset_root=table_asset_root
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if must_exist:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Unable to resolve table asset {table_asset!r}. Tried: {searched}"
        )
    return str(candidates[0])


@dataclass
class ScenarioRecord:
    raw: dict

    @property
    def id(self):
        return str(self.raw["id"])

    @property
    def table_asset(self):
        return self.raw.get("table_asset")


class ScenarioLoader:
    def __init__(self, scenario_file: str):
        self.scenario_file = str(scenario_file)
        self.records = []
        self.records_by_id = {}

        with Path(self.scenario_file).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = ScenarioRecord(json.loads(line))
                self.records.append(record)
                self.records_by_id[record.id] = record

        if not self.records:
            raise ValueError(f"No scenario records found in {self.scenario_file}")

    def list_ids(self):
        return [record.id for record in self.records]

    def list_records(self):
        return [record.raw for record in self.records]

    def get_by_id(self, scenario_id: str):
        return self.records_by_id[str(scenario_id)].raw

    def get_by_ids(self, scenario_ids):
        return [self.get_by_id(scenario_id) for scenario_id in scenario_ids]
