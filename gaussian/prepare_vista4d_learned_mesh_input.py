#!/usr/bin/env python3
"""Adapt a camera-bound learned-mesh orbit to Vista4D conditioning.

Unlike the rejected procedural carrier adapter, this stage only accepts a
Hunyuan multi-view mesh derived from the four reviewed real photos.  It also
requires the exact 49-camera schedule used to render the source frames; the
last frame may not duplicate the first camera.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from gaussian.prepare_four_view_generation import (
    sha256_file,
    validate_generation_input,
)
from gaussian.provenance_quarantine import assert_not_quarantined
from gaussian.record_learned_mesh_orbit_review import ACCEPTED, REVIEW_SCHEMA


INPUT_SCHEMA = "radeon_oneloop.four_view_learned_mesh_texture_orbit.v2"
OUTPUT_SCHEMA = "radeon_oneloop.vista4d_object_conditioning.v1"
DONE_SCHEMA = "radeon_oneloop.vista4d_object_conditioning_done.v1"
FRAME_COUNT = 49
IMAGE_SIZE_WH = (672, 384)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hash_records(root: Path) -> dict[str, str]:
    path = root / "hashes.sha256"
    if not path.is_file():
        raise ValueError("learned-mesh source is missing hashes.sha256")
    records: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relpath = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed source hash line {number}") from exc
        if Path(relpath).is_absolute() or ".." in Path(relpath).parts:
            raise ValueError(f"unsafe source hash path on line {number}")
        records[relpath] = digest
    return records


def _verify_hash_index(root: Path) -> dict[str, str]:
    records = _hash_records(root)
    for relpath, expected in records.items():
        path = root / relpath
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"learned-mesh source hash mismatch: {relpath}")
    return records


def validate_orbit_contract(manifest: dict[str, Any], cameras: dict[str, np.ndarray]) -> None:
    orbit = manifest.get("orbit", {})
    if orbit.get("frames") != FRAME_COUNT or orbit.get("image_size_wh") != list(IMAGE_SIZE_WH):
        raise ValueError("learned-mesh orbit does not match the Vista4D 384p49 contract")
    if orbit.get("endpoint_duplicate") is not False:
        raise ValueError("learned-mesh orbit illegally duplicates the 0/360-degree endpoint")
    if orbit.get("camera_schedule") != "vista4d_unique_49_frame_level_orbit":
        raise ValueError("learned-mesh orbit has an unsupported camera schedule")
    if orbit.get("render_camera_model") != "PINHOLE_OPENCV_fixed_intrinsic":
        raise ValueError("learned-mesh source was not rendered with a fixed pinhole camera")
    expected = np.arange(FRAME_COUNT, dtype=np.float64) * (360.0 / FRAME_COUNT)
    azimuths = np.asarray(cameras.get("azimuth_deg"), dtype=np.float64)
    c2w = np.asarray(cameras.get("cam_c2w"), dtype=np.float64)
    intrinsics = np.asarray(cameras.get("intrinsics"), dtype=np.float64)
    if azimuths.shape != (FRAME_COUNT,) or not np.allclose(
        azimuths, expected, atol=1e-10, rtol=0.0
    ):
        raise ValueError("learned-mesh source images and target azimuths are not exact")
    if c2w.shape != (FRAME_COUNT, 4, 4) or intrinsics.shape != (FRAME_COUNT, 4):
        raise ValueError("learned-mesh target camera arrays have invalid shapes")
    if not np.all(np.isfinite(c2w)) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("learned-mesh target cameras contain non-finite values")


def validate_texture_source(
    root: Path, four_view_input: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = root.resolve()
    four_view_input = four_view_input.resolve()
    four_view_manifest = validate_generation_input(four_view_input)
    manifest_path = root / "manifest.json"
    done_path = root / "DONE"
    if not manifest_path.is_file() or not done_path.is_file():
        raise ValueError("learned-mesh source requires manifest.json and DONE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("only corrected learned-mesh orbit v2 may enter Vista4D")
    if manifest.get("formal") is not False:
        raise ValueError("learned-mesh source must remain nonformal")
    if manifest.get("input", {}).get("inherited_procedural_geometry") is not None:
        raise ValueError("learned-mesh source inherited prohibited procedural geometry")
    expected_four_view = sha256_file(four_view_input / "manifest.json")
    if manifest.get("input", {}).get("four_view_manifest_sha256") != expected_four_view:
        raise ValueError("learned-mesh source does not derive from the selected four real views")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("learned-mesh DONE does not bind its manifest")
    if done.get("hashes_sha256") != sha256_file(root / "hashes.sha256"):
        raise ValueError("learned-mesh DONE does not bind its hash index")
    _verify_hash_index(root)
    camera_path = root / str(manifest["orbit"]["target_cameras_relpath"])
    if sha256_file(camera_path) != manifest["orbit"]["target_cameras_sha256"]:
        raise ValueError("learned-mesh target camera hash mismatch")
    with np.load(camera_path, allow_pickle=False) as stored:
        cameras = {name: np.asarray(stored[name]) for name in stored.files}
    validate_orbit_contract(manifest, cameras)
    return manifest, four_view_manifest, camera_path


def validate_visual_review(
    review_path: Path, texture_root: Path, texture_manifest: dict[str, Any]
) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema_version") != REVIEW_SCHEMA or review.get("decision") != ACCEPTED:
        raise ValueError("learned-mesh orbit requires an accepted external visual review")
    if not all(review.get("checks", {}).values()):
        raise ValueError("learned-mesh visual review checks are incomplete")
    evidence = review.get("evidence", {})
    if evidence.get("texture_manifest_sha256") != sha256_file(texture_root / "manifest.json"):
        raise ValueError("visual review does not bind the selected texture orbit")
    if evidence.get("orbit_contact_sheet_sha256") != sha256_file(
        texture_root / "audit/orbit_contact_sheet.png"
    ):
        raise ValueError("visual review does not bind the selected contact sheet")
    if evidence.get("source_video_sha256") != texture_manifest["orbit"]["source_video_sha256"]:
        raise ValueError("visual review does not bind the selected source video")
    assert_not_quarantined(
        [("learned_mesh_texture_manifest", texture_manifest), ("learned_mesh_visual_review", review)]
    )
    return review


def _write_hashes(root: Path) -> str:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def prepare_conditioning(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - remote generation environment
        raise RuntimeError("imageio and imageio-ffmpeg are required") from exc
    source_root = args.texture_root.resolve()
    four_view_root = args.four_view_input.resolve()
    output = args.output.resolve()
    source, four_view, camera_path = validate_texture_source(source_root, four_view_root)
    review_path = args.review.resolve()
    review = validate_visual_review(review_path, source_root, source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Vista4D conditioning: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        names = (
            "frames",
            "frames_source",
            "alpha_mask_src",
            "dynamic_mask_src",
            "alpha_mask_pc",
            "dynamic_mask_pc",
        )
        for name in names:
            (staging / name).mkdir()
        frames: list[np.ndarray] = []
        support: list[float] = []
        for index in range(FRAME_COUNT):
            frame_source = source_root / "orbit" / "frames" / f"{index:05d}.png"
            alpha_source = source_root / "orbit" / "alpha" / f"{index:05d}.png"
            frame = np.asarray(iio.imread(frame_source), dtype=np.uint8)
            alpha = np.asarray(iio.imread(alpha_source), dtype=np.uint8)
            if frame.shape != (IMAGE_SIZE_WH[1], IMAGE_SIZE_WH[0], 3):
                raise ValueError(f"learned-mesh frame {index} has an invalid shape")
            if alpha.shape != (IMAGE_SIZE_WH[1], IMAGE_SIZE_WH[0]):
                raise ValueError(f"learned-mesh alpha {index} has an invalid shape")
            zero = np.zeros_like(alpha, dtype=np.uint8)
            for folder in ("frames", "frames_source"):
                iio.imwrite(staging / folder / f"{index:05d}.png", frame)
            for folder in ("alpha_mask_src", "alpha_mask_pc"):
                iio.imwrite(staging / folder / f"{index:05d}.png", alpha)
            for folder in ("dynamic_mask_src", "dynamic_mask_pc"):
                iio.imwrite(staging / folder / f"{index:05d}.png", zero)
            frames.append(frame)
            support.append(float(np.mean(alpha >= 128)))
        for name in ("video_src.mp4", "video_pc.mp4"):
            iio.imwrite(
                staging / name,
                np.stack(frames),
                fps=args.fps,
                codec="libx264",
                pixelformat="yuv420p",
            )
        shutil.copy2(camera_path, staging / "cameras_tgt.npz")
        shutil.copy2(
            source_root / "audit" / "orbit_contact_sheet.png",
            staging / "source_contact_sheet.png",
        )
        learned_mesh_sha = source["mesh"]["ply_sha256"]
        manifest = {
            "schema_version": OUTPUT_SCHEMA,
            "created_utc": utc_now(),
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "physical_output": False,
            "host_role": "phi_amd_work_mi300x_nonformal_generation_lab",
            "frames": FRAME_COUNT,
            "image_size_wh": list(IMAGE_SIZE_WH),
            "fps": args.fps,
            "camera": {
                "trajectory": "vista4d_unique_49_frame_level_canonical_orbit",
                "path_topology": "cyclic",
                "endpoint_duplicate": False,
                "distance_m": source["orbit"]["distance_m"],
                "stored_convention": "vista4d_external_c2w",
                "render_camera_model": "PINHOLE_OPENCV_fixed_intrinsic",
                "cameras_relpath": "cameras_tgt.npz",
                "cameras_sha256": sha256_file(staging / "cameras_tgt.npz"),
            },
            "source_video": {
                "role": "four_real_view_projected_Hunyuan_learned_mesh_orbit",
                "relpath": "video_src.mp4",
                "sha256": sha256_file(staging / "video_src.mp4"),
                "surface_carrier": None,
                "real_identity_keyframes": [
                    {
                        "frame_index": index,
                        "view_id": view_id,
                        "role": "conditioning_lineage_only_not_pixel_injection",
                    }
                    for index, view_id in ((0, "front"), (37, "right"), (24, "back"), (12, "left"))
                ],
                "alpha_support_fraction": {
                    "min": float(np.min(support)),
                    "mean": float(np.mean(support)),
                    "max": float(np.max(support)),
                },
            },
            "point_cloud_condition": {
                "role": "same_camera_bound_learned_mesh_render_not_observed_point_cloud",
                "relpath": "video_pc.mp4",
                "sha256": sha256_file(staging / "video_pc.mp4"),
                "dynamic": False,
            },
            "asset": {
                "role": "generated_complete_mesh_prior_aligned_to_four_real_silhouettes",
                "hashes": {"ply": learned_mesh_sha},
                "source_texture_manifest_sha256": sha256_file(source_root / "manifest.json"),
                "four_view_manifest_sha256": sha256_file(four_view_root / "manifest.json"),
                "observed_input_count": len(four_view["observed_inputs"]),
                "inherited_procedural_geometry": None,
            },
            "manual_visual_review": {
                "schema_version": review["schema_version"],
                "decision": review["decision"],
                "accepted_role": "Vista4D_conditioning_only",
                "review_sha256": sha256_file(review_path),
                "checks": review["checks"],
                "known_defects": review["known_defects"],
            },
            "allowed_role": "Vista4D_generated_appearance_pseudoview_conditioning",
            "not_proven": [
                "metric hidden-side geometry",
                "held-out real-view quality",
                "physics collision geometry",
                "single-Radeon reproducibility",
            ],
        }
        (staging / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hashes(staging)
        done = {
            "schema_version": DONE_SCHEMA,
            "stage": "learned_mesh_orbit_to_Vista4D_conditioning",
            "status": "complete",
            "completed_utc": utc_now(),
            "manifest_sha256": sha256_file(staging / "input_manifest.json"),
            "hashes_sha256": hashes_sha,
        }
        (staging / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture-root", type=Path, required=True)
    parser.add_argument("--four-view-input", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = prepare_conditioning(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILED").write_text(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}) + "\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
