#!/usr/bin/env python3
"""Prepare a geometry-free four-view contract for generative Real2Sim.

The only identity inputs accepted by this stage are the four reviewed photos
from the same physical product instance.  It deliberately accepts no mesh,
Gaussian, depth map, or procedural carrier.  Downstream generators may create
a complete prior, but they cannot silently inherit the rejected carrier.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any

import numpy as np


INPUT_SCHEMA_VERSION = "radeon_oneloop.object_asset_manifest.v1"
OUTPUT_SCHEMA_VERSION = "radeon_oneloop.four_view_generation_input.v2"
DONE_SCHEMA_VERSION = "radeon_oneloop.four_view_generation_input_done.v2"
EXPECTED_COORDINATES = {
    "front_axis": "+Y",
    "up_axis": "+Z",
    "viewer_left_axis": "+X",
    "unit": "m",
    "origin": "plush_body_center",
}
VIEW_SPECS = (
    ("anchor_front", "front", 0.0),
    ("anchor_right", "right", -90.0),
    ("anchor_rear", "back", 180.0),
    ("anchor_left", "left", 90.0),
)


class FourViewInputError(ValueError):
    """Raised when reviewed inputs cannot enter the generation mainline."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str, *, field: str) -> Path:
    if not value or "\\" in value:
        raise FourViewInputError(f"{field} must be a non-empty POSIX relative path")
    posix = PurePosixPath(value)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise FourViewInputError(f"{field} must not contain absolute or dot segments")
    return Path(*posix.parts)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FourViewInputError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FourViewInputError(f"{label} must be a JSON object")
    return value


def _parse_hash_index(root: Path) -> dict[str, str]:
    path = root / "hashes.sha256"
    if not path.is_file():
        raise FourViewInputError("reviewed root is missing hashes.sha256")
    records: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relpath = line.split("  ", 1)
        except ValueError as exc:
            raise FourViewInputError(f"malformed hash line {line_number}") from exc
        relative = _relative_path(relpath, field=f"hashes.sha256:{line_number}").as_posix()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise FourViewInputError(f"invalid SHA-256 on line {line_number}")
        if relative in records:
            raise FourViewInputError(f"duplicate hash entry: {relative}")
        records[relative] = digest
    if not records:
        raise FourViewInputError("reviewed hash index is empty")
    return records


def _verify_bound_file(root: Path, relpath: str, expected: str, hashes: dict[str, str]) -> Path:
    relative = _relative_path(relpath, field="reviewed file").as_posix()
    if hashes.get(relative) != expected:
        raise FourViewInputError(f"manifest/hash-index mismatch for {relative}")
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected:
        raise FourViewInputError(f"missing or tampered reviewed file: {relative}")
    return path


def load_reviewed_views(reviewed_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reviewed_root = reviewed_root.resolve()
    manifest_path = reviewed_root / "manifest.json"
    done_path = reviewed_root / "DONE"
    if not manifest_path.is_file() or not done_path.is_file():
        raise FourViewInputError("reviewed root requires manifest.json and DONE")
    manifest = _load_json(manifest_path, label="reviewed manifest")
    done = _load_json(done_path, label="reviewed DONE")
    hashes = _parse_hash_index(reviewed_root)
    manifest_sha = sha256_file(manifest_path)
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise FourViewInputError("unsupported reviewed manifest schema")
    if manifest.get("formal") is not False or manifest.get("redistribution") is not False:
        raise FourViewInputError("reviewed generation inputs must remain nonformal/private")
    if manifest.get("coordinate_convention") != EXPECTED_COORDINATES:
        raise FourViewInputError("reviewed coordinate convention is not canonical")
    if done.get("manifest_sha256") != manifest_sha or hashes.get("manifest.json") != manifest_sha:
        raise FourViewInputError("reviewed DONE/hash index does not bind manifest.json")
    summary = manifest.get("summary", {})
    if summary.get("prepared_count") != 4 or summary.get("generated_count") != 0:
        raise FourViewInputError("mainline requires exactly four prepared observed views")

    by_id = {view.get("id"): view for view in manifest.get("views", []) if isinstance(view, dict)}
    selected: list[dict[str, Any]] = []
    instance_ids: set[str] = set()
    for view_id, generator_label, expected_azimuth in VIEW_SPECS:
        view = by_id.get(view_id)
        if view is None:
            raise FourViewInputError(f"missing required reviewed view: {view_id}")
        roles = set(view.get("roles", []))
        if (
            view.get("provenance") != "observed"
            or view.get("tier") != "A"
            or view.get("prepared") is not True
            or view.get("mask_status") != "reviewed_pass"
            or not {"pose", "photometric", "identity", "generation_input"}.issubset(roles)
        ):
            raise FourViewInputError(f"{view_id} is not an accepted observed generation anchor")
        azimuth = float(view.get("nominal_camera_orbit_deg", {}).get("azimuth", math.nan))
        elevation = float(view.get("nominal_camera_orbit_deg", {}).get("elevation", math.nan))
        if not math.isclose(azimuth, expected_azimuth, abs_tol=1e-6) or not math.isclose(
            elevation, 0.0, abs_tol=1e-6
        ):
            raise FourViewInputError(f"{view_id} nominal camera does not match the four-view contract")
        record = {
            "id": view_id,
            "generator_label": generator_label,
            "azimuth_deg": expected_azimuth,
            "elevation_deg": elevation,
            "instance_id": str(view.get("instance_id")),
        }
        for key in ("image", "neutral_image", "hard_mask", "soft_alpha"):
            item = view.get(key)
            if not isinstance(item, dict) or not isinstance(item.get("relpath"), str):
                raise FourViewInputError(f"{view_id}.{key} is incomplete")
            if not isinstance(item.get("sha256"), str):
                raise FourViewInputError(f"{view_id}.{key} has no SHA-256")
            record[f"{key}_path"] = _verify_bound_file(
                reviewed_root, item["relpath"], item["sha256"], hashes
            )
            record[f"{key}_sha256"] = item["sha256"]
        instance_ids.add(record["instance_id"])
        selected.append(record)
    if len(instance_ids) != 1:
        raise FourViewInputError("the four generation anchors do not show the same physical instance")
    return manifest, selected


def orbit_c2w(azimuth_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    """Return an OpenGL camera-to-world matrix looking at the origin."""
    if not math.isfinite(radius) or radius <= 0:
        raise FourViewInputError("camera radius must be positive and finite")
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    position = np.array(
        [
            radius * math.cos(elevation) * math.sin(azimuth),
            radius * math.cos(elevation) * math.cos(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )
    forward = -position / np.linalg.norm(position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = position
    return matrix


def _read_images(record: dict[str, Any], size: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on real2sim extra
        raise RuntimeError("OpenCV is required to prepare four-view generation inputs") from exc
    rgb_bgr = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    neutral_bgr = cv2.imread(str(record["neutral_image_path"]), cv2.IMREAD_COLOR)
    alpha = cv2.imread(str(record["soft_alpha_path"]), cv2.IMREAD_GRAYSCALE)
    if rgb_bgr is None or neutral_bgr is None or alpha is None:
        raise FourViewInputError(f"cannot decode reviewed pixels for {record['id']}")
    if rgb_bgr.shape[:2] != alpha.shape or neutral_bgr.shape[:2] != alpha.shape:
        raise FourViewInputError(f"RGB/alpha dimensions disagree for {record['id']}")
    interpolation = cv2.INTER_AREA if rgb_bgr.shape[0] > size else cv2.INTER_CUBIC
    rgb_bgr = cv2.resize(rgb_bgr, (size, size), interpolation=interpolation)
    neutral_bgr = cv2.resize(neutral_bgr, (size, size), interpolation=interpolation)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    alpha_f = alpha.astype(np.float32)[..., None] / 255.0
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    neutral = cv2.cvtColor(neutral_bgr, cv2.COLOR_BGR2RGB)
    white = np.full_like(rgb, 255)
    rgb_white = np.rint(rgb * alpha_f + white * (1.0 - alpha_f)).astype(np.uint8)
    neutral_rgba = np.concatenate([neutral, alpha[..., None]], axis=2)
    return rgb_white, neutral_rgba


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
        raise FourViewInputError(f"unsupported image shape for {path.name}: {rgb.shape}")
    code = cv2.COLOR_RGB2BGR if rgb.shape[2] == 3 else cv2.COLOR_RGBA2BGRA
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, code)):
        raise OSError(f"failed to write image: {path}")


def _write_hash_index(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def validate_generation_input(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load_json(root / "manifest.json", label="generation-input manifest")
    done = _load_json(root / "DONE", label="generation-input DONE")
    if manifest.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise FourViewInputError("unsupported generation-input schema")
    policy = manifest.get("source_policy", {})
    if policy.get("geometry_input") is not None or policy.get("procedural_geometry_allowed") is not False:
        raise FourViewInputError("generation input illegally admits an inherited geometry prior")
    if manifest.get("formal") is not False or manifest.get("generated_frames_are_observed") is not False:
        raise FourViewInputError("generation provenance boundary was weakened")
    hashes_path = root / "hashes.sha256"
    if done.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise FourViewInputError("DONE does not bind generation-input manifest")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise FourViewInputError("DONE does not bind generation-input hash index")
    hashes = _parse_hash_index(root)
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise FourViewInputError(f"generation-input hash mismatch: {relative}")
    schedule = manifest.get("target_orbit", {})
    if (
        schedule.get("frame_count") != 49
        or schedule.get("path_topology") != "cyclic"
        or schedule.get("endpoint_duplicate") is not False
    ):
        raise FourViewInputError(
            "generation input must contain the exact cyclic 49-frame orbit without "
            "a duplicated endpoint"
        )
    with np.load(root / str(schedule.get("camera_npz_relpath")), allow_pickle=False) as cameras:
        azimuths = np.asarray(cameras["azimuth_deg"], dtype=np.float64)
    expected = np.arange(49, dtype=np.float64) * (360.0 / 49.0)
    if azimuths.shape != (49,) or not np.allclose(azimuths, expected, atol=1e-5, rtol=0.0):
        raise FourViewInputError("target-camera azimuths do not match the unique Vista4D schedule")
    return manifest


def prepare_generation_input(
    reviewed_root: Path,
    output: Path,
    *,
    image_size: int = 576,
    target_frames: int = 49,
    elevation_deg: float = 0.0,
    camera_radius: float = 2.0,
    horizontal_fov_deg: float = 50.0,
) -> dict[str, Any]:
    if image_size != 576:
        raise FourViewInputError("SEVA v1.1 contract is pinned to 576 x 576")
    if target_frames != 49:
        raise FourViewInputError("Vista4D handoff requires exactly 49 target frames")
    if not 20.0 <= horizontal_fov_deg <= 90.0:
        raise FourViewInputError("horizontal FOV must be in [20, 90] degrees")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite generation input: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    reviewed_manifest, views = load_reviewed_views(reviewed_root)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        anchors_dir = staging / "anchors"
        seva_scene = staging / "seva" / "graffiti_mickey_four_view"
        seva_images = seva_scene / "images"
        hunyuan_dir = staging / "hunyuan3d_2mv"
        for directory in (anchors_dir, seva_images, hunyuan_dir):
            directory.mkdir(parents=True, exist_ok=True)

        source_records = []
        transform_frames: list[dict[str, Any]] = []
        focal = image_size / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
        for index, record in enumerate(views):
            rgb_white, neutral_rgba = _read_images(record, image_size)
            anchor_name = f"{record['generator_label']}.png"
            _write_rgb(anchors_dir / anchor_name, rgb_white)
            _write_rgb(hunyuan_dir / anchor_name, neutral_rgba)
            seva_name = f"input_{record['generator_label']}.png"
            _write_rgb(seva_images / seva_name, rgb_white)
            c2w = orbit_c2w(record["azimuth_deg"], record["elevation_deg"], camera_radius)
            transform_frames.append(
                {
                    "fl_x": focal,
                    "fl_y": focal,
                    "cx": image_size / 2.0,
                    "cy": image_size / 2.0,
                    "w": image_size,
                    "h": image_size,
                    "file_path": f"./images/{seva_name}",
                    "transform_matrix": c2w.tolist(),
                }
            )
            source_records.append(
                {
                    "id": record["id"],
                    "generator_label": record["generator_label"],
                    "azimuth_deg": record["azimuth_deg"],
                    "elevation_deg": record["elevation_deg"],
                    "instance_id": record["instance_id"],
                    "source_hashes": {
                        key: record[f"{key}_sha256"]
                        for key in ("image", "neutral_image", "hard_mask", "soft_alpha")
                    },
                    "prepared_rgb_relpath": f"anchors/{anchor_name}",
                    "hunyuan_rgba_relpath": f"hunyuan3d_2mv/{anchor_name}",
                    "seva_rgb_relpath": f"seva/graffiti_mickey_four_view/images/{seva_name}",
                    "provenance": "observed",
                }
            )

        azimuths = np.arange(target_frames, dtype=np.float64) * (360.0 / target_frames)
        target_c2ws = []
        target_intrinsics = []
        target_schedule = []
        black = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for target_index, azimuth in enumerate(azimuths):
            name = f"target_{target_index:03d}.png"
            _write_rgb(seva_images / name, black)
            c2w = orbit_c2w(float(azimuth), elevation_deg, camera_radius)
            target_c2ws.append(c2w)
            target_intrinsics.append([focal, focal, image_size / 2.0, image_size / 2.0])
            target_schedule.append(
                {
                    "frame_index": target_index,
                    "azimuth_deg": float(azimuth),
                    "elevation_deg": elevation_deg,
                    "camera_radius": camera_radius,
                    "provenance": "generated_target_placeholder",
                }
            )
            transform_frames.append(
                {
                    "fl_x": focal,
                    "fl_y": focal,
                    "cx": image_size / 2.0,
                    "cy": image_size / 2.0,
                    "w": image_size,
                    "h": image_size,
                    "file_path": f"./images/{name}",
                    "transform_matrix": c2w.tolist(),
                }
            )

        transforms = {"orientation_override": "none", "frames": transform_frames}
        (seva_scene / "transforms.json").write_text(
            json.dumps(transforms, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        split = {
            "train_ids": list(range(4)),
            "test_ids": list(range(4, 4 + target_frames)),
        }
        (seva_scene / "train_test_split_4.json").write_text(
            json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            staging / "target_cameras.npz",
            cam_c2w=np.asarray(target_c2ws, dtype=np.float64),
            intrinsics=np.asarray(target_intrinsics, dtype=np.float64),
            azimuth_deg=azimuths.astype(np.float64),
            elevation_deg=np.full(target_frames, elevation_deg, dtype=np.float64),
        )
        (staging / "target_schedule.json").write_text(
            json.dumps(target_schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        reviewed_manifest_sha = sha256_file(reviewed_root.resolve() / "manifest.json")
        manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "asset_name": reviewed_manifest["asset_name"],
            "formal": False,
            "redistribution": False,
            "generated_frames_are_observed": False,
            "eligible_for_heldout_real_metrics": False,
            "source_policy": {
                "identity_source": "exactly_four_reviewed_same_instance_observed_photos",
                "geometry_input": None,
                "procedural_geometry_allowed": False,
                "surface_carrier_allowed": False,
                "reviewed_manifest_sha256": reviewed_manifest_sha,
            },
            "coordinate_convention": EXPECTED_COORDINATES,
            "metric_anchor": reviewed_manifest["metric_anchor"],
            "observed_inputs": source_records,
            "target_orbit": {
                "frame_count": target_frames,
                "image_size_wh": [image_size, image_size],
                "azimuth_start_deg": 0.0,
                "azimuth_step_deg": float(360.0 / target_frames),
                "azimuth_last_deg": float(azimuths[-1]),
                "elevation_deg": elevation_deg,
                "camera_radius_normalized": camera_radius,
                "horizontal_fov_deg": horizontal_fov_deg,
                "path_topology": "cyclic",
                "endpoint_duplicate": False,
                "camera_basis": "nominal_product_orbit_not_photogrammetric_calibration",
                "camera_npz_relpath": "target_cameras.npz",
                "schedule_relpath": "target_schedule.json",
            },
            "generator_contracts": {
                "seva": {
                    "model": "stabilityai/stable-virtual-camera",
                    "version": "1.1",
                    "task": "img2trajvid",
                    "num_inputs": 4,
                    "scene_relpath": "seva/graffiti_mickey_four_view",
                    "role": "primary_camera_controlled_orbit_video_prior",
                },
                "hunyuan3d_2mv": {
                    "model": "tencent/Hunyuan3D-2mv",
                    "subfolder": "hunyuan3d-dit-v2-mv",
                    "input_relpath": "hunyuan3d_2mv",
                    "view_keys": ["front", "left", "back", "right"],
                    "role": "primary_generated_complete_mesh_prior",
                },
                "vista4d": {
                    "role": "downstream_reshooting_and_temporal_consistency_only",
                    "direct_four_still_input_supported": False,
                },
            },
            "release_gates": {
                "requires_real_anchor_reprojection": True,
                "requires_loop_and_temporal_audit": True,
                "requires_generated_region_labels": True,
                "requires_single_radeon_replay_for_formal_claim": True,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hash_index(staging)
        done = {
            "schema_version": DONE_SCHEMA_VERSION,
            "stage": "four_reviewed_views_to_generator_inputs",
            "completed_utc": utc_now(),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "hashes_sha256": hashes_sha,
        }
        (staging / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return validate_generation_input(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=576)
    parser.add_argument("--target-frames", type=int, default=49)
    parser.add_argument("--elevation-deg", type=float, default=0.0)
    parser.add_argument("--camera-radius", type=float, default=2.0)
    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_only:
        result = validate_generation_input(args.output)
    else:
        result = prepare_generation_input(
            args.reviewed_root,
            args.output,
            image_size=args.image_size,
            target_frames=args.target_frames,
            elevation_deg=args.elevation_deg,
            camera_radius=args.camera_radius,
            horizontal_fov_deg=args.horizontal_fov_deg,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
