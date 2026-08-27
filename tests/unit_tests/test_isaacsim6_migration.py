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

"""Tests for the Isaac Sim 6 split-environment migration utilities."""

import importlib.util
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_SCRIPT = REPO_ROOT / "toolkits/isaaclab/migrate_runtime_assets.py"
INSTALL_SCRIPT = REPO_ROOT / "requirements/install_isaacsim6_split.sh"
ISAACLAB_PATCH = REPO_ROOT / "requirements/patches/isaaclab3_local_asset_root.patch"


def load_migration_module():
    """Load the standalone migration script without package side effects."""
    spec = importlib.util.spec_from_file_location(
        "migrate_runtime_assets", MIGRATION_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_asset_migration_filters_and_checksums(tmp_path: Path) -> None:
    """Only runtime files are copied and recorded in the manifest."""
    module = load_migration_module()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    manifest_path = tmp_path / "manifest.json"
    table = source / "SeattleLabTable/color_tables/table_copper.usd"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"usd-data")
    scenario = source / "all_setting/curriculum_nearest100/scenarios.jsonl"
    scenario.parent.mkdir(parents=True)
    scenario.write_text(
        json.dumps({"id": 1, "table_asset": "table_copper.usd"}) + "\n",
        encoding="utf-8",
    )
    embedding = source / "all_setting/embedding/features.npy"
    embedding.parent.mkdir(parents=True)
    embedding.write_bytes(b"excluded")

    manifest = module.migrate(source, destination, manifest_path)

    assert manifest["file_count"] == 2
    assert not (destination / embedding.relative_to(source)).exists()
    assert manifest["files"]["SeattleLabTable/color_tables/table_copper.usd"][
        "sha256"
    ] == module.sha256(table)
    assert json.loads(manifest_path.read_text())["file_count"] == 2


def test_split_installer_has_valid_shell_and_documents_direct_downloads() -> None:
    """The public installer is syntactically valid and exposes its policy."""
    subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], check=True)
    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "proxy variables removed" in result.stdout
    assert "--skip-assets" in result.stdout
    assert ".venv-model312" in result.stdout


def test_split_installer_tracks_local_asset_root_patch() -> None:
    """The ignored Isaac Lab checkout must be reproducible from a tracked patch."""
    installer = INSTALL_SCRIPT.read_text()
    patch = ISAACLAB_PATCH.read_text()

    assert "apply_isaaclab_patch" in installer
    assert "ISAACSIM_ASSET_ROOT" in patch
    assert "asset_root_override.rstrip" in patch


def test_nearest100_configs_have_no_source_repo_absolute_path() -> None:
    """Migrated nearest100 configs must not point back to the old checkout."""
    configs = (
        "isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100.yaml",
        "nearest100_table_copper.yaml",
        "nearest100_table_copper_vlm_lora.yaml",
    )
    old_root = "/RLinf-IsaacLab-Diverse-PPO/RLinf/"
    for config in configs:
        content = (REPO_ROOT / "examples/embodiment/config" / config).read_text()
        assert old_root not in content
        assert "RLINF_ISAACSIM_PYTHON" in content


def test_nearest100_camera_quaternions_are_converted_to_isaaclab3_order() -> None:
    """Scenario JSON uses wxyz, while Isaac Lab 3 camera APIs require xyzw."""
    source_files = (
        REPO_ROOT / "rlinf/envs/isaaclab/custom_events.py",
        REPO_ROOT / "rlinf/envs/isaaclab/venv.py",
    )
    for source_file in source_files:
        content = source_file.read_text()
        assert "ros_quat_wxyz[[1, 2, 3, 0]]" in content


def test_scenario_table_reset_derives_static_table_paths_for_all_envs() -> None:
    """Static Isaac Lab props may expose only env_0 in ``prim_paths``."""
    content = (REPO_ROOT / "rlinf/envs/isaaclab/custom_events.py").read_text()
    assert "def _table_prim_path_for_env" in content
    assert 'return f"{env.scene.env_prim_paths[env_id]}{suffix}"' in content
