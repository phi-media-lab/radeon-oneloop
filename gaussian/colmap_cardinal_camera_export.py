#!/usr/bin/env python3
"""Export four cardinal real-anchor cameras from a COLMAP text dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VIEW_ORDER = ("front", "right", "back", "left")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm((w, x, y, z)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("COLMAP quaternion is invalid")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_colmap_text(root: Path) -> tuple[dict[int, dict], dict[str, dict]]:
    cameras: dict[int, dict] = {}
    for line in (root / "sparse/0/cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        camera_id, model, width, height = int(values[0]), values[1], int(values[2]), int(values[3])
        if model != "PINHOLE" or len(values) != 8:
            raise ValueError("camera export only supports COLMAP PINHOLE")
        fx, fy, cx, cy = map(float, values[4:])
        cameras[camera_id] = {
            "image_size_wh": [width, height],
            "intrinsic_3x3": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        }
    images: dict[str, dict] = {}
    lines = (root / "sparse/0/images.txt").read_text(encoding="utf-8").splitlines()
    for offset in range(0, len(lines)):
        line = lines[offset]
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 10:
            continue
        image_id = int(values[0])
        qvec = np.asarray([float(value) for value in values[1:5]], dtype=np.float64)
        translation = np.asarray([float(value) for value in values[5:8]], dtype=np.float64)
        camera_id = int(values[8])
        name = values[9]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = qvec_to_rotation(qvec)
        transform[:3, 3] = translation
        images[name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "world_to_camera_opencv_4x4": transform.tolist(),
        }
    return cameras, images


def export(dataset: Path, output: Path, mode: str = "cardinal_real") -> dict:
    root = dataset.resolve()
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supported_schemas = {
        "radeon_oneloop.hybrid_pseudoview_colmap_dataset.v1",
        "radeon_oneloop.seva_pseudoview_colmap_dataset.v1",
        "radeon_oneloop.seva_full_geometry_colmap_dataset.v1",
    }
    if manifest.get("schema_version") not in supported_schemas:
        raise ValueError("camera export requires a reviewed pseudo-view dataset")
    cameras, images = parse_colmap_text(root)
    if mode == "cardinal_real":
        selected = [(label, f"real_{label}_w00.png") for label in VIEW_ORDER]
        pose_status = "weak_perspective_equivalent_uncalibrated_real_anchor"
    elif mode == "generated_orbit":
        selected = [(f"orbit_{index:05d}", f"gen_{index:05d}.png") for index in range(49)]
        pose_status = "exact_fixed_pinhole_generated_orbit_camera"
    elif mode == "completed_training_views":
        # Put the real front anchor first because orbit/runtime consumers use
        # the first record only to recover a canonical pinhole intrinsic.
        names = [
            name
            for name in images
            if name != "000_eval_probe_generated.png"
        ]
        names.sort(key=lambda name: (name != "real_front_w00.png", name))
        selected = [(f"training_{index:03d}", name) for index, name in enumerate(names)]
        pose_status = "reviewed_real_weighted_SEVA_completion_camera"
    elif mode == "full_geometry_training_views":
        names = [name for name in images if name != "000_eval_probe_generated.png"]
        names.sort(key=lambda name: (name != "real_front_w00.png", name))
        selected = [(f"training_{index:03d}", name) for index, name in enumerate(names)]
        pose_status = "generated_geometry_hypothesis_exact_SEVA_camera"
    else:
        raise ValueError(f"unsupported camera export mode: {mode}")
    records = []
    for label, name in selected:
        image = images[name]
        camera = cameras[image["camera_id"]]
        records.append(
            {
                "view_id": label,
                "source_image_name": name,
                **camera,
                "world_to_camera_opencv_4x4": image["world_to_camera_opencv_4x4"],
                "pose_status": pose_status,
            }
        )
    value = {
        "schema_version": "radeon_oneloop.hybrid_cardinal_cameras.v1",
        "formal": False,
        "camera_model": "PINHOLE_OPENCV",
        "mode": mode,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "cameras": records,
        "eligible_for_heldout_real_metrics": False,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "cardinal_real",
            "generated_orbit",
            "completed_training_views",
            "full_geometry_training_views",
        ),
        default="cardinal_real",
    )
    args = parser.parse_args()
    print(json.dumps(export(args.dataset, args.output, args.mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
