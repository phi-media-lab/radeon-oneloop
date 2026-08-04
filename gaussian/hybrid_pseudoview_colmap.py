#!/usr/bin/env python3
"""Build a provenance-weighted real+generated COLMAP training workspace.

This exporter never turns generated frames into observations.  The four real
anchors are explicit, repeated training samples; learned-mesh or Vista4D
frames remain low-confidence pseudo-views.  A generated duplicate occupies
VkSplat's sole validation slot so all four unique real anchors remain in
training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from gaussian.align_generated_mesh_four_views import VIEW_ORDER, _fit_projection
from gaussian.audit_vista4d_completion import AUDIT_SCHEMA, read_video, sha256_file
from gaussian.export_observed_initialization import SCHEMA as OBSERVED_INIT_SCHEMA, load_colmap_points
from gaussian.object_colmap_export import rotation_matrix_to_colmap_qvec
from gaussian.prepare_four_view_generation import validate_generation_input
from gaussian.provenance_quarantine import assert_not_quarantined
from gaussian.record_vista4d_completion_review import ACCEPTED, REVIEW_SCHEMA
from gaussian.texture_learned_mesh_four_views import canonical_orbit_extrinsic


SCHEMA = "radeon_oneloop.hybrid_pseudoview_colmap_dataset.v1"
DONE_SCHEMA = "radeon_oneloop.hybrid_pseudoview_colmap_dataset_done.v1"
TEXTURE_SCHEMA = "radeon_oneloop.four_view_learned_mesh_texture_orbit.v2"
PROPOSAL_SCHEMA = "radeon_oneloop.vista4d_object_completion_proposal.v1"
CONDITIONING_SCHEMA = "radeon_oneloop.vista4d_object_conditioning.v1"
IMAGE_SIZE_WH = (672, 384)
VIEW_AZIMUTHS = {"front": 0.0, "right": 270.0, "back": 180.0, "left": 90.0}


class HybridDatasetError(ValueError):
    """Raised when generated and observed evidence cannot be mixed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HybridDatasetError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HybridDatasetError(f"{label} must be a JSON object")
    return value


def verify_hash_index(root: Path) -> dict[str, str]:
    index = root / "hashes.sha256"
    if not index.is_file():
        raise HybridDatasetError(f"missing hash index: {index}")
    records: dict[str, str] = {}
    for line_number, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relpath = line.split("  ", 1)
        except ValueError as exc:
            raise HybridDatasetError(f"malformed hash line {line_number}: {index}") from exc
        relative = Path(relpath)
        if relative.is_absolute() or ".." in relative.parts:
            raise HybridDatasetError(f"unsafe hash path on line {line_number}: {relpath}")
        path = root / relative
        if len(digest) != 64 or not path.is_file() or sha256_file(path) != digest:
            raise HybridDatasetError(f"hash mismatch: {root.name}/{relpath}")
        records[relpath] = digest
    return records


def recover_vista4d_w2c(stored_c2w: np.ndarray) -> np.ndarray:
    """Recover OpenCV world-to-camera matrices from Vista4D storage convention."""

    values = np.asarray(stored_c2w, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise HybridDatasetError("Vista4D camera array must have shape N x 4 x 4")
    if not np.all(np.isfinite(values)):
        raise HybridDatasetError("Vista4D camera array contains non-finite values")
    conversion = np.diag([-1.0, -1.0, 1.0, 1.0])
    return np.stack([np.linalg.inv(conversion @ matrix) for matrix in values])


def resize_and_composite_anchor(
    rgb: np.ndarray, alpha: np.ndarray, target_size_wh: tuple[int, int] = IMAGE_SIZE_WH
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Aspect-preserving resize and white-pad of one real masked photograph."""

    import cv2

    image = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(alpha, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise HybridDatasetError("real anchor RGB and alpha shapes differ")
    target_width, target_height = target_size_wh
    height, width = mask.shape
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_rgb = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_alpha = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    region = canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width]
    weight = resized_alpha.astype(np.float32)[..., None] / 255.0
    region[:] = np.rint(resized_rgb * weight + 255.0 * (1.0 - weight)).astype(np.uint8)
    canvas_mask[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = (
        resized_alpha
    )
    return canvas, canvas_mask, {
        "source_size_wh": [width, height],
        "target_size_wh": [target_width, target_height],
        "uniform_scale": float(scale),
        "offset_xy": [offset_x, offset_y],
        "pixel_warp": "aspect_preserving_resize_plus_white_padding",
    }


def weak_perspective_equivalent_camera(
    vertices: np.ndarray,
    mask: np.ndarray,
    label: str,
    *,
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Approximate an uncalibrated product photo by a distant pinhole camera."""

    if label not in VIEW_AZIMUTHS or not math.isfinite(distance_m) or distance_m <= 0:
        raise HybridDatasetError("invalid weak-perspective camera request")
    _, fit, _ = _fit_projection(np.asarray(vertices, dtype=np.float64), mask >= 128, label)
    pixels_per_m = float(fit["pixels_per_raw_unit"])
    focal_px = pixels_per_m * distance_m
    principal_x = float(fit["target_center_x_px"] - fit["source_center_u_raw"] * pixels_per_m)
    principal_y = float(fit["target_center_y_px"] - fit["source_center_v_raw"] * pixels_per_m)
    intrinsic = np.asarray(
        [[focal_px, 0.0, principal_x], [0.0, focal_px, principal_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    w2c = canonical_orbit_extrinsic(VIEW_AZIMUTHS[label], distance_m=distance_m)
    report = {
        "model": "PINHOLE_weak_perspective_equivalent_for_uncalibrated_product_photo",
        "distance_m": distance_m,
        "pixels_per_m": pixels_per_m,
        "focal_px": focal_px,
        "principal_xy": [principal_x, principal_y],
        "camera_bound_observation": False,
        "purpose": "real_anchor_supervision_with_explicit_pose_uncertainty",
    }
    return intrinsic, w2c, report


def _colmap_camera_line(camera_id: int, intrinsic: np.ndarray) -> str:
    width, height = IMAGE_SIZE_WH
    return (
        f"{camera_id} PINHOLE {width} {height} {intrinsic[0, 0]:.17g} "
        f"{intrinsic[1, 1]:.17g} {intrinsic[0, 2]:.17g} {intrinsic[1, 2]:.17g}\n"
    )


def _colmap_image_line(
    image_id: int, camera_id: int, name: str, world_to_camera: np.ndarray
) -> str:
    transform = np.asarray(world_to_camera, dtype=np.float64)
    qvec = rotation_matrix_to_colmap_qvec(transform[:3, :3])
    values = [*qvec.tolist(), *transform[:3, 3].tolist()]
    return (
        f"{image_id} "
        + " ".join(f"{value:.17g}" for value in values)
        + f" {camera_id} {name}\n\n"
    )


def _load_mask_bundle(root: Path) -> np.ndarray:
    import cv2

    paths = sorted((root / "generated_masks").glob("*.png"))
    if len(paths) != 49:
        raise HybridDatasetError(f"audit contains {len(paths)} generated masks, expected 49")
    masks = []
    for path in paths:
        value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if value is None or value.shape != (IMAGE_SIZE_WH[1], IMAGE_SIZE_WH[0]):
            raise HybridDatasetError(f"invalid generated mask: {path}")
        masks.append(value)
    return np.stack(masks)


def _load_texture_orbit(root: Path) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    frames = []
    masks = []
    for index in range(49):
        frame_path = root / "orbit/frames" / f"{index:05d}.png"
        mask_path = root / "orbit/alpha" / f"{index:05d}.png"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or frame.shape != (IMAGE_SIZE_WH[1], IMAGE_SIZE_WH[0], 3):
            raise HybridDatasetError(f"invalid learned-mesh orbit frame: {frame_path}")
        if mask is None or mask.shape != (IMAGE_SIZE_WH[1], IMAGE_SIZE_WH[0]):
            raise HybridDatasetError(f"invalid learned-mesh orbit mask: {mask_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks.append(mask)
    return np.stack(frames), np.stack(masks)


def _load_observed_initialization(root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest_path = root / "manifest.json"
    done_path = root / "DONE"
    if not manifest_path.is_file() or not done_path.is_file():
        raise HybridDatasetError("observed visual-hull initialization is incomplete")
    manifest = _load_json(manifest_path, "observed initialization manifest")
    done = _load_json(done_path, "observed initialization DONE")
    if manifest.get("schema_version") != OBSERVED_INIT_SCHEMA:
        raise HybridDatasetError("hybrid dataset requires the observed visual-hull initializer")
    provenance = manifest.get("provenance", {})
    if any(
        provenance.get(key) is not False
        for key in ("generated_geometry", "generated_views", "learned_depth", "secondary_accelerator_artifacts")
    ):
        raise HybridDatasetError("observed initialization contains generated or learned geometry")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise HybridDatasetError("observed initialization DONE does not bind manifest")
    if done.get("hashes_sha256") != sha256_file(root / "hashes.sha256"):
        raise HybridDatasetError("observed initialization DONE does not bind hashes")
    points_path = root / manifest["points"]["relpath"]
    if sha256_file(points_path) != manifest["points"]["sha256"]:
        raise HybridDatasetError("observed initialization point hash mismatch")
    vertices, colors = load_colmap_points(points_path)
    if len(vertices) != manifest["points"]["count"]:
        raise HybridDatasetError("observed initialization point count mismatch")
    return manifest, vertices, colors


def _write_hashes(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    four_view_root = args.four_view_input.resolve()
    texture_root = args.texture_root.resolve()
    conditioning_root = args.conditioning.resolve()
    observed_initialization_root = args.observed_initialization.resolve()
    proposal_root = args.proposal_run.resolve() if args.proposal_run is not None else None
    audit_root = args.audit.resolve() if args.audit is not None else None
    review_path = args.review.resolve() if args.review is not None else None
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.real_repeat < 1 or args.max_points < 1000 or args.real_camera_distance_m < 0.5:
        raise HybridDatasetError("invalid sampling, point, or real-camera configuration")

    four_view = validate_generation_input(four_view_root)
    texture = _load_json(texture_root / "manifest.json", "texture manifest")
    texture_done = _load_json(texture_root / "DONE", "texture DONE")
    if texture.get("schema_version") != TEXTURE_SCHEMA:
        raise HybridDatasetError("only corrected learned-mesh texture v2 is supported")
    if texture_done.get("manifest_sha256") != sha256_file(texture_root / "manifest.json"):
        raise HybridDatasetError("texture DONE does not bind manifest")
    verify_hash_index(texture_root)
    mesh_path = texture_root / texture["mesh"]["ply_relpath"]
    if sha256_file(mesh_path) != texture["mesh"]["ply_sha256"]:
        raise HybridDatasetError("texture mesh hash mismatch")

    conditioning = _load_json(conditioning_root / "input_manifest.json", "conditioning")
    conditioning_done = _load_json(conditioning_root / "DONE", "conditioning DONE")
    if conditioning.get("schema_version") != CONDITIONING_SCHEMA:
        raise HybridDatasetError("unexpected Vista4D conditioning schema")
    verify_hash_index(conditioning_root)
    if conditioning_done.get("manifest_sha256") != sha256_file(
        conditioning_root / "input_manifest.json"
    ):
        raise HybridDatasetError("conditioning DONE does not bind manifest")

    assert_not_quarantined(
        [
            ("four_view_manifest", four_view),
            ("texture_manifest", texture),
            ("conditioning_manifest", conditioning),
        ]
    )
    observed_initialization, vertices, colors = _load_observed_initialization(
        observed_initialization_root
    )
    assert_not_quarantined(
        [("observed_visual_hull_initialization", observed_initialization)]
    )

    camera_path = conditioning_root / conditioning["camera"]["cameras_relpath"]
    if sha256_file(camera_path) != conditioning["camera"]["cameras_sha256"]:
        raise HybridDatasetError("conditioning camera hash mismatch")
    with np.load(camera_path, allow_pickle=False) as stored:
        stored_c2w = np.asarray(stored["cam_c2w"], dtype=np.float64)
        packed_intrinsics = np.asarray(stored["intrinsics"], dtype=np.float64)
        azimuths = np.asarray(stored["azimuth_deg"], dtype=np.float64)
    if stored_c2w.shape != (49, 4, 4) or packed_intrinsics.shape != (49, 4):
        raise HybridDatasetError("conditioning cameras violate the 49-frame contract")
    generated_w2c = recover_vista4d_w2c(stored_c2w)
    if args.source_mode == "vista4d":
        if proposal_root is None or audit_root is None or review_path is None:
            raise HybridDatasetError("Vista4D source mode requires proposal, audit, and review")
        proposal = _load_json(proposal_root / "manifest.json", "proposal")
        proposal_done = _load_json(proposal_root / "DONE", "proposal DONE")
        if proposal.get("schema_version") != PROPOSAL_SCHEMA:
            raise HybridDatasetError("unexpected Vista4D proposal schema")
        if proposal_done.get("manifest_sha256") != sha256_file(proposal_root / "manifest.json"):
            raise HybridDatasetError("proposal DONE does not bind manifest")
        verify_hash_index(proposal_root)
        if proposal["conditioning"]["manifest_sha256"] != sha256_file(
            conditioning_root / "input_manifest.json"
        ):
            raise HybridDatasetError("proposal does not derive from selected conditioning")
        generated_path = (
            proposal_root / "inference" / f"video_seed={proposal['model']['seed']}.mp4"
        )
        if sha256_file(generated_path) != proposal["generated_video_sha256"]:
            raise HybridDatasetError("generated video hash mismatch")
        audit = _load_json(audit_root / "metrics.json", "completion audit")
        audit_done = _load_json(audit_root / "DONE", "completion audit DONE")
        if audit.get("schema_version") != AUDIT_SCHEMA:
            raise HybridDatasetError("hybrid dataset requires mask-bound completion audit v2")
        if audit_done.get("metrics_sha256") != sha256_file(audit_root / "metrics.json"):
            raise HybridDatasetError("audit DONE does not bind metrics")
        if audit_done.get("hashes_sha256") != sha256_file(audit_root / "hashes.sha256"):
            raise HybridDatasetError("audit DONE does not bind hashes")
        verify_hash_index(audit_root)
        review = _load_json(review_path, "completion human review")
        assert_not_quarantined(
            [
                ("vista4d_proposal", proposal),
                ("vista4d_audit", audit),
                ("vista4d_human_review", review),
            ]
        )
        if review.get("schema_version") != REVIEW_SCHEMA or review.get("decision") != ACCEPTED:
            raise HybridDatasetError("Vista4D proposal was not accepted for pseudo-view training")
        if review.get("audit_metrics_sha256") != sha256_file(audit_root / "metrics.json"):
            raise HybridDatasetError("human review does not bind selected audit")
        if not all(review.get("checks", {}).values()):
            raise HybridDatasetError("human review identity checks are incomplete")
        generated_frames = read_video(generated_path)
        generated_masks = _load_mask_bundle(audit_root)
        generated_provenance = "generated_low_confidence_vista4d_pseudoview"
        generated_mask_source = "audited_inferred_foreground_v2"
        generated_source_role = "Vista4D_generated_appearance_completion"
        lineage_extra = {
            "proposal_manifest_sha256": sha256_file(proposal_root / "manifest.json"),
            "audit_metrics_sha256": sha256_file(audit_root / "metrics.json"),
            "human_review_sha256": sha256_file(review_path),
        }
    elif args.source_mode == "learned_mesh_orbit":
        if any(value is not None for value in (proposal_root, audit_root, review_path)):
            raise HybridDatasetError("learned-mesh source mode must not accept Vista4D evidence")
        manual = conditioning.get("manual_visual_review", {})
        if manual.get("decision") != "accepted_conditioning_only":
            raise HybridDatasetError("learned-mesh orbit lacks its explicit visual conditioning review")
        if not all(manual.get("checks", {}).values()) or not manual.get("review_sha256"):
            raise HybridDatasetError("learned-mesh orbit review is incomplete or unbound")
        if conditioning.get("asset", {}).get("source_texture_manifest_sha256") != sha256_file(
            texture_root / "manifest.json"
        ):
            raise HybridDatasetError("conditioning review does not bind selected texture orbit")
        generated_frames, generated_masks = _load_texture_orbit(texture_root)
        generated_provenance = "generated_low_confidence_camera_bound_learned_mesh_render"
        generated_mask_source = "exact_learned_mesh_raster_alpha"
        generated_source_role = "Hunyuan_learned_mesh_orbit_with_four_real_projected_colors"
        lineage_extra = {
            "proposal_manifest_sha256": None,
            "audit_metrics_sha256": None,
            "human_review_sha256": manual["review_sha256"],
        }
    else:  # pragma: no cover - argparse enforces the choice
        raise HybridDatasetError(f"unsupported source mode: {args.source_mode}")
    if generated_frames.shape != (49, IMAGE_SIZE_WH[1], IMAGE_SIZE_WH[0], 3):
        raise HybridDatasetError("generated video shape violates the 384p49 contract")

    valid = np.all(np.isfinite(vertices), axis=1)
    candidates = np.flatnonzero(valid)
    if len(candidates) < 1000:
        raise HybridDatasetError("observed visual hull has too few finite initialization vertices")
    if len(candidates) > args.max_points:
        generator = np.random.default_rng(args.sample_seed)
        candidates = np.sort(generator.choice(candidates, args.max_points, replace=False))
    vertices = vertices[candidates]
    colors = colors[candidates]

    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    try:
        image_dir = staging / "images"
        mask_dir = staging / "masks"
        sparse_dir = staging / "sparse/0"
        for directory in (image_dir, mask_dir, sparse_dir):
            directory.mkdir(parents=True)

        entries: list[dict[str, Any]] = []
        for index in range(49):
            intrinsic = np.asarray(
                [
                    [packed_intrinsics[index, 0], 0.0, packed_intrinsics[index, 2]],
                    [0.0, packed_intrinsics[index, 1], packed_intrinsics[index, 3]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            entries.append(
                {
                    "name": f"gen_{index:05d}.png",
                    "rgb": generated_frames[index],
                    "mask": generated_masks[index],
                    "intrinsic": intrinsic,
                    "w2c": generated_w2c[index],
                    "provenance": generated_provenance,
                    "source_index": index,
                    "azimuth_deg": float(azimuths[index]),
                    "sampling_duplicate": False,
                }
            )
        entries.append(
            {
                **entries[0],
                "name": "000_eval_probe_generated_00000.png",
                "provenance": "generated_eval_probe_duplicate_not_training_evidence",
                "sampling_duplicate": True,
            }
        )

        real_records = []
        by_label = {item["generator_label"]: item for item in four_view["observed_inputs"]}
        for label in VIEW_ORDER:
            item = by_label[label]
            rgb_bgr = cv2.imread(str(four_view_root / item["prepared_rgb_relpath"]), cv2.IMREAD_COLOR)
            rgba_bgra = cv2.imread(str(four_view_root / item["hunyuan_rgba_relpath"]), cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or rgba_bgra is None or rgba_bgra.shape[2] != 4:
                raise HybridDatasetError(f"cannot read real anchor: {label}")
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            prepared, prepared_mask, image_transform = resize_and_composite_anchor(
                rgb, rgba_bgra[..., 3]
            )
            intrinsic, w2c, camera_report = weak_perspective_equivalent_camera(
                vertices,
                prepared_mask,
                label,
                distance_m=args.real_camera_distance_m,
            )
            for repeat in range(args.real_repeat):
                entries.append(
                    {
                        "name": f"real_{label}_w{repeat:02d}.png",
                        "rgb": prepared,
                        "mask": prepared_mask,
                        "intrinsic": intrinsic,
                        "w2c": w2c,
                        "provenance": "observed_real_anchor_sampling_duplicate",
                        "source_index": None,
                        "azimuth_deg": VIEW_AZIMUTHS[label],
                        "sampling_duplicate": repeat > 0,
                    }
                )
            real_records.append(
                {
                    "view": label,
                    "source_id": item["id"],
                    "source_hashes": item["source_hashes"],
                    "training_repetitions": args.real_repeat,
                    "image_transform": image_transform,
                    "camera": camera_report,
                }
            )

        entries.sort(key=lambda item: item["name"])
        camera_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"]
        image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
        exported = []
        for image_id, entry in enumerate(entries, 1):
            name = entry["name"]
            if not cv2.imwrite(
                str(image_dir / name), cv2.cvtColor(entry["rgb"], cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"failed to write {name}")
            if not cv2.imwrite(str(mask_dir / name), np.asarray(entry["mask"], dtype=np.uint8)):
                raise RuntimeError(f"failed to write mask {name}")
            camera_lines.append(_colmap_camera_line(image_id, entry["intrinsic"]))
            image_lines.append(_colmap_image_line(image_id, image_id, name, entry["w2c"]))
            exported.append(
                {
                    key: entry[key]
                    for key in (
                        "name",
                        "provenance",
                        "source_index",
                        "azimuth_deg",
                        "sampling_duplicate",
                    )
                }
            )
        (sparse_dir / "cameras.txt").write_text("".join(camera_lines), encoding="utf-8")
        (sparse_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")
        point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"]
        for point_id, (point, color) in enumerate(zip(vertices, colors, strict=True), 1):
            point_lines.append(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
        (sparse_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")

        real_instances = 4 * args.real_repeat
        generated_training = 49
        training_total = real_instances + generated_training
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "formal": False,
            "eligible_for_heldout_real_metrics": False,
            "asset_name": four_view["asset_name"],
            "lineage": {
                "four_view_manifest_sha256": sha256_file(four_view_root / "manifest.json"),
                "texture_manifest_sha256": sha256_file(texture_root / "manifest.json"),
                "conditioning_manifest_sha256": sha256_file(conditioning_root / "input_manifest.json"),
                "observed_initialization_manifest_sha256": sha256_file(
                    observed_initialization_root / "manifest.json"
                ),
                **lineage_extra,
                "inherited_procedural_geometry": None,
            },
            "sampling": {
                "real_unique_views": 4,
                "real_repetitions_per_view": args.real_repeat,
                "real_training_instances": real_instances,
                "generated_unique_training_views": generated_training,
                "generated_eval_probe_duplicates": 1,
                "training_instances_total": training_total,
                "nominal_real_sampling_probability": real_instances / training_total,
                "mechanism": "explicit_filename_duplicates_for_uniform_VkSplat_sampler",
                "interpretation": "loss_weight_approximation_not_independent_real_evidence",
            },
            "vksplat_split_contract": {
                "recommended_eval_interval": len(entries),
                "lexicographically_first_eval_image": entries[0]["name"],
                "all_unique_real_anchors_train": True,
                "eval_probe_is_generated_duplicate": True,
            },
            "generated_views": {
                "count": 49,
                "source_mode": args.source_mode,
                "source_role": generated_source_role,
                "camera_model": "exact_fixed_pinhole_bound_generated_orbit_track",
                "mask_source": generated_mask_source,
                "confidence": "low",
                "allowed_role": "appearance_completion_training_only",
            },
            "real_views": real_records,
            "initial_points": {
                "count": int(len(vertices)),
                "source": "observed_real_mask_CPU_visual_hull",
                "source_points_sha256": observed_initialization["points"]["sha256"],
                "source_manifest_sha256": sha256_file(
                    observed_initialization_root / "manifest.json"
                ),
                "sample_seed": args.sample_seed,
                "generated_geometry_prior": False,
                "observed_visual_hull_prior": True,
                "metric_truth": False,
            },
            "images": exported,
            "required_training_profile": {
                "freeze_geometry": True,
                "disable_refinement": True,
                "keep_output_as_generated_fill_layer": True,
            },
            "provenance_boundary": [
                "Generated frames are pseudo-views, never observed or held-out-real evidence.",
                "Repeated real files change sampling frequency, not evidence count.",
                "Real product-photo cameras are disclosed weak-perspective approximations.",
                "GS geometry initializes only from the observed CPU visual hull and must remain frozen.",
                "The generated source supplies pseudo-view appearance but no initialization points.",
            ],
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hashes(staging)
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "done_nonformal_hybrid_dataset",
            "manifest_sha256": sha256_file(staging / "dataset_manifest.json"),
            "hashes_sha256": hashes_sha,
            "completed_utc": utc_now(),
        }
        (staging / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return manifest
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps(
                {
                    "schema_version": "radeon_oneloop.hybrid_pseudoview_colmap_failure.v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_utc": utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failed = output.with_name(f"{output.name}.FAILED")
        if not failed.exists():
            os.replace(staging, failed)
        else:
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mode", choices=("vista4d", "learned_mesh_orbit"), required=True
    )
    parser.add_argument("--four-view-input", type=Path, required=True)
    parser.add_argument("--texture-root", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--observed-initialization", type=Path, required=True)
    parser.add_argument("--proposal-run", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-repeat", type=int, default=12)
    parser.add_argument("--real-camera-distance-m", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    result = build_dataset(parse_args())
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "sampling": result["sampling"],
                "initial_points": result["initial_points"],
                "vksplat_split_contract": result["vksplat_split_contract"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
