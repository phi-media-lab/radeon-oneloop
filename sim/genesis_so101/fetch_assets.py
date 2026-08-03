#!/usr/bin/env python3
"""Fetch the pinned Apache-2.0 SO-101 MJCF asset with hash verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


COMMIT = "1b74d9fcaaa8e0514bf504545f8f3555a044e0bf"
BASE_URL = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{COMMIT}/Simulation/SO101"
FILES = {
    "so101_new_calib.xml": (13315, "7be7999f63520aee5b7925169a69599de57b06569a758c88fbecdc5e4e8aa7d3"),
    "assets/base_motor_holder_so101_v1.stl": (1877084, "8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424"),
    "assets/base_so101_v2.stl": (471584, "bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d"),
    "assets/motor_holder_so101_base_v1.stl": (1129384, "31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b"),
    "assets/motor_holder_so101_wrist_v1.stl": (1052184, "887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011"),
    "assets/moving_jaw_so101_v1.stl": (1413584, "785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d"),
    "assets/rotation_pitch_so101_v1.stl": (883684, "9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34"),
    "assets/sts3215_03a_no_horn_v1.stl": (865884, "75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887"),
    "assets/sts3215_03a_v1.stl": (954084, "a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0"),
    "assets/under_arm_so101_v1.stl": (1975884, "d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71"),
    "assets/upper_arm_so101_v1.stl": (1303484, "475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f"),
    "assets/waveshare_mounting_plate_so101_v2.stl": (62784, "e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9"),
    "assets/wrist_roll_follower_so101_v1.stl": (1439884, "4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4"),
    "assets/wrist_roll_pitch_so101_v2.stl": (2699784, "6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(root: Path) -> dict[str, object]:
    records = []
    for relative, (expected_size, expected_hash) in FILES.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        valid = destination.is_file() and destination.stat().st_size == expected_size and digest(destination) == expected_hash
        if not valid:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(f"{BASE_URL}/{relative}", timeout=120) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if temporary.stat().st_size != expected_size or digest(temporary) != expected_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"asset verification failed: {relative}")
            temporary.replace(destination)
        records.append({"path": relative, "bytes": expected_size, "sha256": expected_hash})
    manifest = {
        "schema_version": "radeon_oneloop.so101_assets.v1",
        "source": "TheRobotStudio/SO-ARM100",
        "commit": COMMIT,
        "license": "Apache-2.0",
        "files": records,
    }
    (root / "ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("assets") / "so101")
    args = parser.parse_args()
    print(json.dumps(fetch(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
