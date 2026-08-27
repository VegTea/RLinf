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

"""Extract only the local assets required by the Isaac Lab Stack Cube task."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PREFIXES = (
    "Assets/Isaac/6.0/Isaac/Props/Blocks/",
    "Assets/Isaac/6.0/Isaac/Props/Mounts/SeattleLabTable/",
    "Assets/Isaac/6.0/Isaac/IsaacLab/Robots/FrankaEmika/",
    "Assets/Isaac/6.0/Isaac/Environments/Grid/",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Combined Isaac Sim asset ZIP")
    parser.add_argument("destination", type=Path, help="Asset-pack root to create")
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        names = [name for name in archive.namelist() if name.startswith(PREFIXES)]
        if not names:
            raise RuntimeError(f"No Stack Cube assets found in {args.archive}")
        for name in names:
            archive.extract(name, args.destination)
    print(f"Extracted {len(names)} entries to {args.destination}")


if __name__ == "__main__":
    main()
