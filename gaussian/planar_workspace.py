#!/usr/bin/env python3
"""Recover the planar HIL workspace from the visible yellow target square.

The wrist sequence is dominated by a tabletop plane, which is degenerate for
incremental SfM but ideal for target-plane homographies.  This module detects
and tracks the target corners, self-calibrates a pinhole camera from repeated
views of the square, and median-composites a rectified static orthomosaic.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return four points clockwise, starting at image-space top-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    first = ordered[1] - ordered[0]
    second = ordered[2] - ordered[1]
    cross = first[0] * second[1] - first[1] * second[0]
    if cross < 0.0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def track_quad(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Keep physical corner identity continuous across a square's rotations."""
    previous = np.asarray(previous, dtype=np.float32).reshape(4, 2)
    current = order_quad(current)
    candidates = []
    for base in (current, current[[0, 3, 2, 1]]):
        candidates.extend(np.roll(base, shift, axis=0) for shift in range(4))
    return min(
        candidates,
        key=lambda value: float(np.square(value - previous).sum()),
    )


def detect_yellow_quad(
    image: np.ndarray,
    *,
    min_area: float = 2_000.0,
    min_solidity: float = 0.88,
) -> tuple[np.ndarray | None, dict[str, float]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray((15, 70, 55), dtype=np.uint8),
        np.asarray((45, 255, 255), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"area_px2": 0.0, "solidity": 0.0}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0.0 else 0.0
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for epsilon in (0.02, 0.025, 0.03, 0.035, 0.04, 0.05):
        candidate = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(candidate) == 4:
            quad = order_quad(candidate.reshape(4, 2))
            break
    metrics = {"area_px2": area, "solidity": solidity}
    if quad is None or area < min_area or solidity < min_solidity:
        return None, metrics
    side_lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if float(side_lengths.min()) < 20.0:
        return None, metrics
    return quad, metrics


def _calibrate(
    detections: Sequence[dict[str, object]], image_size: tuple[int, int]
) -> tuple[dict[str, object], list[np.ndarray]]:
    object_square = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    object_points = [object_square.copy() for _ in detections]
    image_points = [
        np.asarray(item["corners_px"], dtype=np.float32).reshape(4, 1, 2)
        for item in detections
    ]
    width, height = image_size
    camera = np.asarray(
        ((500.0, 0.0, width / 2.0), (0.0, 500.0, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    distortion = np.zeros((5, 1), dtype=np.float64)
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, camera, distortion, rotations, translations = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        camera,
        distortion,
        flags=flags,
    )
    camera_to_target = []
    errors = []
    for object_value, image_value, rotation, translation in zip(
        object_points, image_points, rotations, translations, strict=True
    ):
        projected, _ = cv2.projectPoints(
            object_value, rotation, translation, camera, distortion
        )
        errors.append(
            float(np.linalg.norm(projected.reshape(4, 2) - image_value.reshape(4, 2), axis=1).mean())
        )
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        target_to_camera = np.eye(4, dtype=np.float64)
        target_to_camera[:3, :3] = rotation_matrix
        target_to_camera[:3, 3] = translation.reshape(3)
        camera_to_target.append(np.linalg.inv(target_to_camera))
    report = {
        "model": "PINHOLE_ZERO_DISTORTION",
        "image_size_wh": [width, height],
        "camera_matrix": camera.tolist(),
        "distortion": distortion.reshape(-1).tolist(),
        "rms_reprojection_px": float(rms),
        "mean_view_error_px": float(np.mean(errors)),
        "p95_view_error_px": float(np.percentile(errors, 95)),
        "target_side_units": 1.0,
        "metric_scale_status": "pending_robot_kinematic_alignment",
    }
    return report, camera_to_target


def _continuous_square_poses(
    detections: Sequence[dict[str, object]], camera_matrix: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Resolve planar IPPE pose ambiguity using within-segment continuity."""
    object_square = np.asarray(
        (
            (-0.5, 0.5, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, -0.5, 0.0),
            (-0.5, -0.5, 0.0),
        ),
        dtype=np.float32,
    )
    distortion = np.zeros(5, dtype=np.float64)
    poses = []
    errors = []
    previous = None
    previous_segment = None
    for detection in detections:
        segment = int(detection["track_segment"])
        if segment != previous_segment:
            previous = None
            previous_segment = segment
        image_points = np.asarray(detection["corners_px"], dtype=np.float32)
        result = cv2.solvePnPGeneric(
            object_square,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not result[0]:
            raise RuntimeError("IPPE failed for a validated yellow-square detection")
        candidates = []
        for rotation_vector, translation_vector in zip(result[1], result[2], strict=True):
            if float(translation_vector.reshape(3)[2]) <= 0.0:
                continue
            rotation, _ = cv2.Rodrigues(rotation_vector)
            target_to_camera = np.eye(4, dtype=np.float64)
            target_to_camera[:3, :3] = rotation
            target_to_camera[:3, 3] = translation_vector.reshape(3)
            camera_to_target = np.linalg.inv(target_to_camera)
            projected, _ = cv2.projectPoints(
                object_square,
                rotation_vector,
                translation_vector,
                camera_matrix,
                distortion,
            )
            reprojection = float(
                np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1).mean()
            )
            score = reprojection
            if previous is not None:
                relative = np.linalg.inv(previous) @ camera_to_target
                cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                score += 0.25 * math.degrees(math.acos(float(cosine)))
                score += 2.0 * float(np.linalg.norm(relative[:3, 3]))
            candidates.append((score, camera_to_target, reprojection))
        if not candidates:
            raise RuntimeError("IPPE returned no positive-depth square pose")
        _, pose, error = min(candidates, key=lambda value: value[0])
        poses.append(pose)
        errors.append(error)
        previous = pose
    return poses, np.asarray(errors, dtype=np.float64)


def _orthomosaic(
    detections: Sequence[dict[str, object]],
    *,
    workspace: Path,
    output: Path,
    canvas_size: int,
    target_pixels: int,
    max_frames: int,
) -> dict[str, object]:
    selection = np.unique(
        np.linspace(0, len(detections) - 1, min(max_frames, len(detections))).astype(int)
    )
    half = target_pixels / 2.0
    center = canvas_size / 2.0
    destination = np.asarray(
        (
            (center - half, center - half),
            (center + half, center - half),
            (center + half, center + half),
            (center - half, center + half),
        ),
        dtype=np.float32,
    )
    warped_images = []
    warped_masks = []
    for index in selection:
        item = detections[int(index)]
        image = cv2.imread(str(workspace / str(item["image"])), cv2.IMREAD_COLOR)
        corners = np.asarray(item["corners_px"], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(corners, destination)
        warped_images.append(
            cv2.warpPerspective(image, transform, (canvas_size, canvas_size))
        )
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_masks.append(
            cv2.warpPerspective(
                source_mask,
                transform,
                (canvas_size, canvas_size),
                flags=cv2.INTER_NEAREST,
            )
            > 0
        )
    images = np.stack(warped_images)
    masks = np.stack(warped_masks)
    coverage = masks.sum(axis=0)
    mosaic = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    tile_rows = 32
    for start in range(0, canvas_size, tile_rows):
        stop = min(start + tile_rows, canvas_size)
        values = images[:, start:stop].astype(np.float32)
        values[~masks[:, start:stop, :, None].repeat(3, axis=3)] = np.nan
        with np.errstate(all="ignore"):
            tile = np.nanmedian(values, axis=0)
        mosaic[start:stop] = np.nan_to_num(tile, nan=0.0).astype(np.uint8)
    coverage_image = np.clip(255.0 * coverage / max(len(selection), 1), 0, 255).astype(
        np.uint8
    )
    cv2.imwrite(str(output / "workspace_orthomosaic.png"), mosaic)
    cv2.imwrite(str(output / "workspace_coverage.png"), coverage_image)
    central = coverage[
        int(center - half) : int(center + half),
        int(center - half) : int(center + half),
    ]
    return {
        "frames_composited": int(len(selection)),
        "canvas_size_px": canvas_size,
        "target_side_px": target_pixels,
        "target_median_coverage": float(np.median(central)),
        "target_min_coverage": int(central.min()),
        "nonzero_canvas_fraction": float(np.count_nonzero(coverage) / coverage.size),
    }


def reconstruct(args: argparse.Namespace) -> dict[str, object]:
    workspace = args.workspace.resolve()
    output = workspace / args.output_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir()
    frame_records = {
        item["image"]: item
        for item in (
            json.loads(line)
            for line in (workspace / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    image_paths = sorted((workspace / "images").glob("*.jpg"))
    detections = []
    previous = None
    previous_frame_index = None
    track_segment = -1
    image_size = None
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {path}")
        image_size = (image.shape[1], image.shape[0])
        quad, quality = detect_yellow_quad(
            image, min_area=args.min_area, min_solidity=args.min_solidity
        )
        if quad is None:
            continue
        source_record = frame_records[path.relative_to(workspace).as_posix()]
        frame_index = int(source_record["frame_index"])
        if (
            previous is None
            or previous_frame_index is None
            or frame_index - previous_frame_index > args.max_track_gap_frames
        ):
            track_segment += 1
            quad = order_quad(quad)
        else:
            quad = track_quad(previous, quad)
            jump = float(np.linalg.norm(quad - previous, axis=1).max())
            if jump > args.max_corner_jump:
                track_segment += 1
                quad = order_quad(quad)
        previous = quad
        previous_frame_index = frame_index
        detections.append(
            {
                **source_record,
                "track_segment": track_segment,
                "corners_px": quad.tolist(),
                **quality,
            }
        )
    if image_size is None or len(detections) < args.min_detections:
        raise RuntimeError(
            f"only {len(detections)} target detections; minimum is {args.min_detections}"
        )
    calibration, _ = _calibrate(detections, image_size)
    camera_poses, ippe_errors = _continuous_square_poses(
        detections, np.asarray(calibration["camera_matrix"], dtype=np.float64)
    )
    calibration["pose_method"] = "IPPE_SQUARE_TEMPORAL_BRANCH_SELECTION"
    calibration["ippe_mean_reprojection_px"] = float(np.mean(ippe_errors))
    calibration["ippe_p95_reprojection_px"] = float(np.percentile(ippe_errors, 95))
    np.savez_compressed(
        output / "camera_poses_unit_target.npz",
        camera_to_target=np.stack(camera_poses),
        episode_index=np.asarray([item["episode_index"] for item in detections]),
        frame_index=np.asarray([item["frame_index"] for item in detections]),
        timestamp_s=np.asarray([item["timestamp_s"] for item in detections]),
        track_segment=np.asarray([item["track_segment"] for item in detections]),
        pose_reprojection_error_px=ippe_errors,
    )
    with (output / "target_detections.jsonl").open("w", encoding="utf-8") as stream:
        for item in detections:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    mosaic = _orthomosaic(
        detections,
        workspace=workspace,
        output=output,
        canvas_size=args.canvas_size,
        target_pixels=args.target_pixels,
        max_frames=args.mosaic_max_frames,
    )
    report = {
        "schema_version": "radeon_oneloop.hil_planar_workspace.v1",
        "formal": False,
        "method": "tracked_yellow_square_homographies_and_temporal_median",
        "input_images": len(image_paths),
        "target_detections": len(detections),
        "track_segments": track_segment + 1,
        "detection_fraction": len(detections) / len(image_paths),
        "calibration": calibration,
        "orthomosaic": mosaic,
        "limitations": [
            "target square side is unitless until SO-101 kinematic alignment",
            "off-plane geometry is excluded from the planar workspace model",
        ],
    }
    metrics = output / "metrics.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashed = [
        metrics,
        output / "camera_poses_unit_target.npz",
        output / "target_detections.jsonl",
        output / "workspace_orthomosaic.png",
        output / "workspace_coverage.png",
    ]
    (output / "hashes.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}" for path in sorted(hashed)
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "DONE").touch()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-name", default="planar")
    parser.add_argument("--min-area", type=float, default=2_000.0)
    parser.add_argument("--min-solidity", type=float, default=0.88)
    parser.add_argument("--min-detections", type=int, default=20)
    parser.add_argument("--max-corner-jump", type=float, default=120.0)
    parser.add_argument("--max-track-gap-frames", type=int, default=15)
    parser.add_argument("--canvas-size", type=int, default=768)
    parser.add_argument("--target-pixels", type=int, default=320)
    parser.add_argument("--mosaic-max-frames", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = reconstruct(args)
    except Exception:
        args.workspace.mkdir(parents=True, exist_ok=True)
        (args.workspace / "PLANAR_FAILED").touch()
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
