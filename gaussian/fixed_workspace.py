#!/usr/bin/env python3
"""Recover the static workspace visible to the fixed HIL front camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_image_crop(image: np.ndarray, threshold: int = 8) -> tuple[int, int, int, int]:
    """Return x, y, width, height after removing uniform letterbox bands."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows = np.mean(gray > threshold, axis=1) > 0.25
    columns = np.mean(gray > threshold, axis=0) > 0.25
    row_indexes = np.flatnonzero(rows)
    column_indexes = np.flatnonzero(columns)
    if row_indexes.size == 0 or column_indexes.size == 0:
        raise ValueError("image has no active region")
    left, right = int(column_indexes[0]), int(column_indexes[-1] + 1)
    top, bottom = int(row_indexes[0]), int(row_indexes[-1] + 1)
    return left, top, right - left, bottom - top


def detect_target_quads(image: np.ndarray, min_area: float = 800.0) -> list[dict[str, object]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray((15, 65, 55), dtype=np.uint8),
        np.asarray((45, 255, 255), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0.0 or area / hull_area < 0.85:
            continue
        perimeter = cv2.arcLength(hull, True)
        quad = cv2.approxPolyDP(hull, 0.035 * perimeter, True)
        if len(quad) != 4:
            continue
        corners = quad.reshape(4, 2).astype(np.float32)
        center = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        corners = corners[np.argsort(angles)]
        targets.append(
            {
                "center_px": center.tolist(),
                "corners_px": corners.tolist(),
                "yellow_area_px2": area,
                "solidity": area / hull_area,
            }
        )
    return sorted(targets, key=lambda item: float(item["center_px"][0]))


def _temporal_statistics(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count, height, width, _ = len(images), *images[0].shape
    median = np.zeros((height, width, 3), dtype=np.uint8)
    variation = np.zeros((height, width), dtype=np.float32)
    tile_rows = 32
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        tile = np.stack([image[start:stop] for image in images]).astype(np.float32)
        tile_median = np.median(tile, axis=0)
        median[start:stop] = tile_median.astype(np.uint8)
        variation[start:stop] = np.median(
            np.abs(tile - tile_median[None, ...]), axis=(0, 3)
        )
    return median, variation


def _table_candidate_background(
    images: list[np.ndarray], fallback: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select low-saturation, mid-value samples likely to be bare gray table."""
    height, width, _ = images[0].shape
    background = np.zeros_like(fallback)
    coverage = np.zeros((height, width), dtype=np.uint16)
    tile_rows = 24
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        tile = np.stack([image[start:stop] for image in images])
        hsv = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for image in tile])
        valid = (hsv[..., 1] < 45) & (hsv[..., 2] >= 45) & (hsv[..., 2] <= 190)
        coverage[start:stop] = valid.sum(axis=0)
        values = tile.astype(np.float32)
        values[~valid[..., None].repeat(3, axis=3)] = np.nan
        with np.errstate(all="ignore"):
            estimate = np.nanmedian(values, axis=0)
        missing = ~np.isfinite(estimate[..., 0])
        estimate[missing] = fallback[start:stop][missing]
        background[start:stop] = estimate.astype(np.uint8)
    return background, coverage


def _expanded_quad(target: dict[str, object], scale: float) -> np.ndarray:
    corners = np.asarray(target["corners_px"], dtype=np.float32)
    center = corners.mean(axis=0)
    return center + scale * (corners - center)


def _clean_table_texture(
    images: list[np.ndarray],
    median: np.ndarray,
    variation: np.ndarray,
    targets: list[dict[str, object]],
    crop: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    background, candidate_coverage = _table_candidate_background(images, median)
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    invalid = (
        (candidate_coverage < 5)
        | (hsv[..., 1] > 65)
        | (hsv[..., 2] < 40)
        | (hsv[..., 2] > 200)
    )
    preserve_targets = np.zeros(invalid.shape, dtype=np.uint8)
    for target in targets:
        cv2.fillConvexPoly(
            preserve_targets, _expanded_quad(target, 1.35).astype(np.int32), 255
        )
    invalid = cv2.dilate(invalid.astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
    invalid[preserve_targets > 0] = 0
    cleaned = cv2.inpaint(background, invalid, 7.0, cv2.INPAINT_TELEA)

    left, top, width, height = crop
    yy, xx = np.mgrid[: median.shape[0], : median.shape[1]]
    xn = 2.0 * (xx / max(median.shape[1] - 1, 1)) - 1.0
    yn = 2.0 * (yy / max(median.shape[0] - 1, 1)) - 1.0
    design = np.stack(
        (np.ones_like(xn), xn, yn, xn * yn, xn * xn, yn * yn), axis=-1
    )
    background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    fit_mask = (
        (candidate_coverage >= max(20, len(images) // 20))
        & (variation < 10.0)
        & (background_hsv[..., 1] < 45)
        & (background_hsv[..., 2] >= 45)
        & (background_hsv[..., 2] <= 190)
        & (xx >= left)
        & (xx < left + width)
        & (yy >= top)
        & (yy < top + height)
        & (preserve_targets == 0)
    )
    fit_indexes = np.flatnonzero(fit_mask)
    if fit_indexes.size < 1_000:
        raise RuntimeError("too few stable table pixels for the smooth simulator texture")
    if fit_indexes.size > 80_000:
        fit_indexes = fit_indexes[
            np.linspace(0, fit_indexes.size - 1, 80_000).astype(int)
        ]
    x_fit = design.reshape(-1, 6)[fit_indexes]
    y_fit = background.reshape(-1, 3)[fit_indexes].astype(np.float64)
    keep = np.ones(len(fit_indexes), dtype=bool)
    coefficients = None
    residual = None
    for _ in range(4):
        coefficients, _, _, _ = np.linalg.lstsq(x_fit[keep], y_fit[keep], rcond=None)
        residual = np.linalg.norm(x_fit @ coefficients - y_fit, axis=1)
        keep = residual <= np.percentile(residual, 80)
    assert coefficients is not None and residual is not None
    smooth = np.clip(design @ coefficients, 0, 255).astype(np.uint8)

    # The right target is unobstructed in the reviewed handover layout. Reuse
    # that real patch to repair the center marker hidden by the repeated arm
    # trajectory, while preserving each target's measured projective shape.
    source = targets[-1]
    source_corners = np.asarray(source["corners_px"], dtype=np.float32)
    source_region = np.zeros(invalid.shape, dtype=np.uint8)
    cv2.fillConvexPoly(source_region, _expanded_quad(source, 1.35).astype(np.int32), 255)
    for target in targets:
        destination_corners = np.asarray(target["corners_px"], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source_corners, destination_corners)
        patch = cv2.warpPerspective(median, transform, (median.shape[1], median.shape[0]))
        patch_mask = cv2.warpPerspective(
            source_region,
            transform,
            (median.shape[1], median.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        cleaned[patch_mask > 0] = patch[patch_mask > 0]
        smooth[patch_mask > 0] = patch[patch_mask > 0]
    return cleaned, smooth, candidate_coverage, {
        "candidate_coverage_median": float(np.median(candidate_coverage)),
        "candidate_coverage_p05": float(np.percentile(candidate_coverage, 5)),
        "inpainted_fraction": float(np.count_nonzero(invalid) / invalid.size),
        "smooth_fit_pixels": int(keep.sum()),
        "smooth_fit_p80_residual_bgr": float(np.percentile(residual, 80)),
    }


def _rectify_for_genesis(
    texture: np.ndarray,
    targets: list[dict[str, object]],
    *,
    table_width_m: float,
    table_depth_m: float,
    target_spacing_m: float,
    target_side_m: float,
    target_forward_y_m: float,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    output_width, output_height = output_size
    source_points = []
    destination_points = []
    destination_quads = []
    world_centers = (
        (target_spacing_m, target_forward_y_m),
        (0.0, target_forward_y_m),
        (-target_spacing_m, target_forward_y_m),
    )
    for target, (world_x, world_y) in zip(targets, world_centers, strict=True):
        source_points.extend(np.asarray(target["corners_px"], dtype=np.float32))
        center_x = (table_width_m / 2.0 - world_x) / table_width_m * output_width
        center_y = (world_y + table_depth_m / 2.0) / table_depth_m * output_height
        half_x = 0.5 * target_side_m / table_width_m * output_width
        half_y = 0.5 * target_side_m / table_depth_m * output_height
        destination_quad = np.asarray(
            (
                (center_x - half_x, center_y - half_y),
                (center_x + half_x, center_y - half_y),
                (center_x + half_x, center_y + half_y),
                (center_x - half_x, center_y + half_y),
            ),
            dtype=np.float32,
        )
        destination_quads.append(destination_quad)
        destination_points.extend(destination_quad)
    transform, _ = cv2.findHomography(
        np.asarray(source_points, dtype=np.float32),
        np.asarray(destination_points, dtype=np.float32),
        method=0,
    )
    if transform is None:
        raise RuntimeError("failed to rectify the three fixed-camera targets")
    rectified = cv2.warpPerspective(texture, transform, output_size)
    observed = cv2.warpPerspective(
        np.full(texture.shape[:2], 255, dtype=np.uint8),
        transform,
        output_size,
        flags=cv2.INTER_NEAREST,
    )
    if not np.any(observed):
        raise RuntimeError("rectified table texture has no observed pixels")
    fill_color = np.median(rectified[observed > 0], axis=0).astype(np.uint8)
    clean_rectified = np.empty_like(rectified)
    clean_rectified[:] = fill_color
    for destination_quad in destination_quads:
        center = destination_quad.mean(axis=0)
        patch_quad = center + 1.45 * (destination_quad - center)
        patch_mask = np.zeros(observed.shape, dtype=np.uint8)
        cv2.fillConvexPoly(patch_mask, patch_quad.astype(np.int32), 255)
        clean_rectified[patch_mask > 0] = rectified[patch_mask > 0]
    rectified = clean_rectified
    projected = cv2.perspectiveTransform(
        np.asarray(source_points, dtype=np.float32).reshape(-1, 1, 2), transform
    ).reshape(-1, 2)
    residual = np.linalg.norm(
        projected - np.asarray(destination_points, dtype=np.float32), axis=1
    )
    return rectified, {
        "table_size_m": [table_width_m, table_depth_m],
        "texture_size_px": [output_width, output_height],
        "target_centers_world_xy_m": [list(value) for value in world_centers],
        "target_side_m": target_side_m,
        "homography_mean_residual_px": float(np.mean(residual)),
        "homography_max_residual_px": float(np.max(residual)),
        "observed_texture_fraction": float(np.count_nonzero(observed) / observed.size),
        "unobserved_fill_bgr": fill_color.tolist(),
        "scale_status": "provisional_from_SO-101_base_separation",
    }


def calibrate_front_camera(
    targets: list[dict[str, object]],
    *,
    image_size: tuple[int, int],
    world_centers_xy_m: list[list[float]],
    target_side_m: float,
    table_surface_z_m: float,
    initial_focal_px: float = 700.0,
) -> dict[str, object]:
    """Fit a P0 fixed-camera view from three coplanar square targets.

    The dataset has no intrinsic or extrinsic calibration.  This fit therefore
    fixes principal point, aspect ratio, and distortion, and estimates only one
    focal length plus the planar camera pose.  Metric scale comes from the
    provisional SO-101 scene layout, so this is a view-alignment calibration,
    not a metrology-grade camera calibration.
    """
    if len(targets) != len(world_centers_xy_m) or len(targets) < 3:
        raise ValueError("camera calibration requires matching data for at least 3 targets")
    width, height = image_size
    half = target_side_m / 2.0
    object_points: list[list[float]] = []
    image_points: list[list[float]] = []
    for target, (center_x, center_y) in zip(
        targets, world_centers_xy_m, strict=True
    ):
        corners = np.asarray(target["corners_px"], dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError("each target must contain 4 ordered image corners")
        # Image left is Genesis +X and image top is Genesis -Y for this fixed
        # operator-side camera. detect_target_quads orders TL, TR, BR, BL.
        object_points.extend(
            (
                (center_x + half, center_y - half, table_surface_z_m),
                (center_x - half, center_y - half, table_surface_z_m),
                (center_x - half, center_y + half, table_surface_z_m),
                (center_x + half, center_y + half, table_surface_z_m),
            )
        )
        image_points.extend(corners.tolist())
    object_array = np.asarray(object_points, dtype=np.float32)
    image_array = np.asarray(image_points, dtype=np.float32)
    intrinsic = np.asarray(
        (
            (initial_focal_px, 0.0, width / 2.0),
            (0.0, initial_focal_px, height / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, intrinsic, distortion, rotation_vectors, translation_vectors = (
        cv2.calibrateCamera(
            [object_array],
            [image_array],
            image_size,
            intrinsic,
            None,
            flags=flags,
        )
    )
    rotation = cv2.Rodrigues(rotation_vectors[0])[0]
    translation = translation_vectors[0]
    camera_position = (-rotation.T @ translation).reshape(3)
    camera_forward = rotation.T @ np.asarray((0.0, 0.0, 1.0))
    camera_up = rotation.T @ np.asarray((0.0, -1.0, 0.0))
    lookat = camera_position + camera_forward
    projected = cv2.projectPoints(
        object_array,
        rotation_vectors[0],
        translation,
        intrinsic,
        distortion,
    )[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - image_array, axis=1)
    vertical_fov_deg = 2.0 * math.degrees(
        math.atan((height / 2.0) / float(intrinsic[1, 1]))
    )
    geometrically_plausible = bool(
        camera_position[2] > table_surface_z_m
        and 20.0 <= vertical_fov_deg <= 100.0
        and camera_forward[2] < 0.0
    )
    p95_error = float(np.percentile(errors, 95))
    accepted = geometrically_plausible and p95_error <= 20.0
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = camera_position
    return {
        "schema_version": "radeon_oneloop.hil_front_camera_calibration.v1",
        "formal": False,
        "method": "single_plane_three_square_view_alignment",
        "status": "accepted_p0_view_alignment" if accepted else "rejected",
        "accepted": accepted,
        "image_size_px": [width, height],
        "intrinsic_matrix_px": intrinsic.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "genesis_camera": {
            "position_m": camera_position.tolist(),
            "lookat_m": lookat.tolist(),
            "up": camera_up.tolist(),
            "vertical_fov_deg": vertical_fov_deg,
        },
        "fit": {
            "rms_px": float(rms),
            "median_reprojection_error_px": float(np.median(errors)),
            "p95_reprojection_error_px": p95_error,
            "max_reprojection_error_px": float(np.max(errors)),
            "correspondences": len(errors),
            "quality_gate_p95_px": 20.0,
            "geometrically_plausible": geometrically_plausible,
        },
        "world_assumptions": {
            "target_centers_xy_m": world_centers_xy_m,
            "target_side_m": target_side_m,
            "table_surface_z_m": table_surface_z_m,
            "scale_status": "provisional_from_SO-101_base_separation",
        },
        "limitations": [
            "single-plane calibration cannot independently validate metric scale",
            "principal point, square pixels, and zero distortion are fixed assumptions",
            "yellow target boundaries are soft image contours rather than surveyed corners",
        ],
    }


def reconstruct(args: argparse.Namespace) -> dict[str, object]:
    workspace = args.workspace.resolve()
    output = workspace / args.output_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir()
    paths = sorted((workspace / "images").glob("*.jpg"))
    if len(paths) < args.min_images:
        raise RuntimeError(f"only {len(paths)} images; minimum is {args.min_images}")
    indexes = np.unique(
        np.linspace(0, len(paths) - 1, min(args.max_images, len(paths))).astype(int)
    )
    selected_paths = [paths[int(index)] for index in indexes]
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in selected_paths]
    if any(image is None for image in images):
        raise RuntimeError("one or more front-camera images could not be decoded")
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise RuntimeError(f"front-camera image shapes differ: {shapes}")
    median, variation = _temporal_statistics(images)
    crop = active_image_crop(median)
    left, top, width, height = crop
    targets = detect_target_quads(median)
    if len(targets) < args.min_targets:
        raise RuntimeError(
            f"only {len(targets)} yellow targets detected; minimum is {args.min_targets}"
        )
    variation_image = np.clip(255.0 * variation / max(args.variation_scale, 1e-6), 0, 255).astype(
        np.uint8
    )
    cleaned, sim_texture, candidate_coverage, cleaning = _clean_table_texture(
        images, median, variation, targets, crop
    )
    rectified, rectification = _rectify_for_genesis(
        sim_texture,
        targets,
        table_width_m=args.table_width_m,
        table_depth_m=args.table_depth_m,
        target_spacing_m=args.target_spacing_m,
        target_side_m=args.target_side_m,
        target_forward_y_m=args.target_forward_y_m,
        output_size=(args.rectified_width, args.rectified_height),
    )
    front_camera_calibration = calibrate_front_camera(
        targets[:3],
        image_size=(median.shape[1], median.shape[0]),
        world_centers_xy_m=rectification["target_centers_world_xy_m"],
        target_side_m=args.target_side_m,
        table_surface_z_m=args.table_surface_z_m,
        initial_focal_px=args.initial_focal_px,
    )
    candidate_coverage_image = np.clip(
        255.0 * candidate_coverage / max(len(images), 1), 0, 255
    ).astype(np.uint8)
    cv2.imwrite(str(output / "front_background_median.png"), median)
    cv2.imwrite(str(output / "front_background_variation.png"), variation_image)
    cv2.imwrite(str(output / "front_background_clean.png"), cleaned)
    cv2.imwrite(
        str(output / "front_sim_table_texture.png"),
        sim_texture[top : top + height, left : left + width],
    )
    cv2.imwrite(str(output / "genesis_table_texture.png"), rectified)
    cv2.imwrite(
        str(output / "front_background_candidate_coverage.png"),
        candidate_coverage_image,
    )
    cv2.imwrite(
        str(output / "front_background_active.png"),
        median[top : top + height, left : left + width],
    )
    calibration_path = output / "front_camera_calibration.json"
    calibration_path.write_text(
        json.dumps(front_camera_calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "radeon_oneloop.hil_fixed_front_workspace.v1",
        "formal": False,
        "method": "multi_episode_temporal_median",
        "input_images": len(paths),
        "images_composited": len(images),
        "active_crop_xywh": list(crop),
        "targets": targets,
        "variation": {
            "median_abs_deviation_px": float(np.median(variation)),
            "p95_abs_deviation_px": float(np.percentile(variation, 95)),
            "scale_for_visualization": args.variation_scale,
        },
        "cleaning": cleaning,
        "genesis_rectification": rectification,
        "front_camera_calibration": front_camera_calibration,
        "limitations": [
            "fixed robot bases can remain in the median background",
            "metric scale remains provisional until a surveyed target is captured",
        ],
    }
    metrics = output / "metrics.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashed = [
        metrics,
        output / "front_background_median.png",
        output / "front_background_variation.png",
        output / "front_background_clean.png",
        output / "front_sim_table_texture.png",
        output / "genesis_table_texture.png",
        output / "front_background_candidate_coverage.png",
        output / "front_background_active.png",
        calibration_path,
    ]
    (output / "hashes.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in sorted(hashed))
        + "\n",
        encoding="utf-8",
    )
    (output / "DONE").touch()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-name", default="fixed_front")
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=600)
    parser.add_argument("--min-targets", type=int, default=3)
    parser.add_argument("--variation-scale", type=float, default=40.0)
    parser.add_argument("--table-width-m", type=float, default=1.2)
    parser.add_argument("--table-depth-m", type=float, default=0.8)
    parser.add_argument("--target-spacing-m", type=float, default=0.2)
    parser.add_argument("--target-side-m", type=float, default=0.08)
    parser.add_argument("--target-forward-y-m", type=float, default=-0.26)
    parser.add_argument("--table-surface-z-m", type=float, default=0.4105)
    parser.add_argument("--initial-focal-px", type=float, default=700.0)
    parser.add_argument("--rectified-width", type=int, default=600)
    parser.add_argument("--rectified-height", type=int, default=400)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = reconstruct(args)
    except Exception:
        args.workspace.mkdir(parents=True, exist_ok=True)
        (args.workspace / "FIXED_FRONT_FAILED").touch()
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
