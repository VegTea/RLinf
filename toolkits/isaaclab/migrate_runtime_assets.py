#!/usr/bin/env python3
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

"""Migrate and checksum RLinf's runtime Isaac Lab scenario assets."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXCLUDED_PARTS = {"embedding", "logs", "log", "cache", "caches", ".thumbs"}
SCENARIO_SUFFIXES = {".json", ".jsonl"}
ASSET_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".mdl",
    ".mtl",
    ".png",
    ".tga",
    ".usda",
    ".usdc",
    ".usd",
}


def should_copy(relative_path: Path) -> bool:
    """Return whether a source-relative asset belongs in the runtime bundle."""
    if set(relative_path.parts) & EXCLUDED_PARTS:
        return False
    if relative_path.parts[0] == "all_setting":
        return relative_path.suffix.lower() in SCENARIO_SUFFIXES
    if relative_path.parts[0] in {"SeattleLabTable", "Materials"}:
        return relative_path.suffix.lower() in ASSET_SUFFIXES
    return False


def sha256(path: Path) -> str:
    """Calculate a file SHA-256 without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_json(path: Path) -> list[str]:
    """Validate JSON/JSONL and return referenced table asset names."""
    table_assets = []
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            records = (json.loads(line) for line in stream if line.strip())
        else:
            value = json.load(stream)
            records = value if isinstance(value, list) else [value]
        for record in records:
            if isinstance(record, dict) and record.get("table_asset"):
                table_assets.append(str(record["table_asset"]))
    return table_assets


def migrate(source: Path, destination: Path, manifest_path: Path) -> dict:
    """Copy the selected runtime bundle and write its deterministic manifest."""
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination asset roots must be different")
    selected = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and should_copy(path.relative_to(source))
    ]
    files = {}
    referenced_tables = set()
    for source_path in selected:
        relative = source_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source_path)
        if not target.exists() or target.stat().st_size != source_path.stat().st_size:
            shutil.copy2(source_path, target)
        target_hash = sha256(target)
        if target_hash != source_hash:
            raise RuntimeError(f"Checksum mismatch after copying {relative}")
        if target.suffix.lower() in SCENARIO_SUFFIXES:
            referenced_tables.update(validate_json(target))
        files[relative.as_posix()] = {
            "sha256": source_hash,
            "size": source_path.stat().st_size,
        }

    table_root = destination / "SeattleLabTable" / "color_tables"
    missing_tables = sorted(
        table
        for table in referenced_tables
        if not (Path(table).is_absolute() and Path(table).exists())
        and not (table_root / table).exists()
        and not (table_root / Path(table).name).exists()
    )
    if missing_tables:
        preview = ", ".join(missing_tables[:20])
        raise FileNotFoundError(f"Scenario files reference missing tables: {preview}")

    manifest = {
        "format_version": 1,
        "source_root_name": source.name,
        "destination_root_name": destination.name,
        "file_count": len(files),
        "total_size": sum(item["size"] for item in files.values()),
        "referenced_table_assets": sorted(referenced_tables),
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the runtime-asset migration."""
    args = parse_args()
    manifest = migrate(args.source, args.destination, args.manifest)
    print(
        f"Migrated {manifest['file_count']} files "
        f"({manifest['total_size']} bytes) to {args.destination}"
    )


if __name__ == "__main__":
    main()
