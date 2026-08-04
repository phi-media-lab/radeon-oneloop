#!/usr/bin/env python3
"""Render a continuous object-centric orbit of the pinned Gaussian asset."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .gaussian_appearance import (
    PinholeCamera,
    VkSplatAppearanceRenderer,
    nonformal_candidate_asset,
    observed_core_asset,
)


def canonical_orbit_extrinsic(
    azimuth_deg: float, *, distance_m: float = 0.3
) -> np.ndarray:
    """Return canonical-object to OpenCV-camera for a level azimuth orbit.

    Azimuth zero is the accepted front camera at ``(0, +distance, 0)``.
    Positive azimuth follows the frozen front/right/rear/left camera order.
    The image vertical axis remains aligned with canonical ``-Z``.
    """

    if not math.isfinite(azimuth_deg) or not math.isfinite(distance_m):
        raise ValueError("orbit parameters must be finite")
    if distance_m <= 0.0:
        raise ValueError("orbit distance must be positive")
    angle = math.radians(azimuth_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        (
            (-cosine, -sine, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (sine, -cosine, 0.0, distance_m),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def scaled_intrinsic(
    intrinsic_3x3: np.ndarray,
    source_size_wh: tuple[int, int],
    output_size_wh: tuple[int, int],
) -> np.ndarray:
    """Scale a pinhole intrinsic matrix to a different raster size."""

    intrinsic = np.asarray(intrinsic_3x3, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_3x3 must be 3x3")
    source_width, source_height = source_size_wh
    output_width, output_height = output_size_wh
    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scaled = intrinsic.copy()
    scaled[0] *= output_width / source_width
    scaled[1] *= output_height / source_height
    return scaled


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _contact_sheet(frames: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not frames:
        raise ValueError("contact sheet requires at least one frame")
    height, width, channels = frames[0].shape
    if channels != 3 or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("contact sheet frames must share one HxWx3 shape")
    rows = math.ceil(len(frames) / columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height,
              column * width : (column + 1) * width] = frame
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--background", type=float, default=0.125)
    parser.add_argument("--alpha-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--candidate-nonformal",
        action="store_true",
        help="Audit a self-bound formal=false candidate instead of the pinned default.",
    )
    args = parser.parse_args()
    if args.frames < 12 or args.frames % 4:
        raise ValueError("frames must be a multiple of four and at least 12")
    if min(args.width, args.height) <= 0:
        raise ValueError("output dimensions must be positive")
    if not 1.0 <= args.fps <= 60.0:
        raise ValueError("fps must be between 1 and 60")
    if not 0.0 <= args.background <= 1.0:
        raise ValueError("background must be in [0, 1]")
    if not 0.0 < args.alpha_threshold < 1.0:
        raise ValueError("alpha-threshold must be in (0, 1)")
    args.output.mkdir(parents=True, exist_ok=False)
    frame_dir = args.output / "frames"
    frame_dir.mkdir()
    import imageio.v3 as iio

    asset = (
        nonformal_candidate_asset(args.asset_root)
        if args.candidate_nonformal
        else observed_core_asset(args.asset_root)
    )
    asset_audit = asset.validate()
    camera_document = json.loads(asset.cameras_path.read_text(encoding="utf-8"))
    front = camera_document["cameras"][0]
    source_width, source_height = (int(value) for value in front["image_size_wh"])
    intrinsic = scaled_intrinsic(
        np.asarray(front["intrinsic_3x3"], dtype=np.float64),
        (source_width, source_height),
        (args.width, args.height),
    )

    renderer = VkSplatAppearanceRenderer(asset, args.vksplat_root)
    rendered: list[np.ndarray] = []
    records = []
    render_times_ms: list[float] = []
    try:
        # Include the repeated 360-degree endpoint so cycle closure is measured
        # directly. The duplicate endpoint is excluded from the video.
        for index in range(args.frames + 1):
            azimuth_deg = 360.0 * index / args.frames
            camera = PinholeCamera(
                width=args.width,
                height=args.height,
                intrinsic_3x3=intrinsic,
                camera_from_object_opencv_4x4=canonical_orbit_extrinsic(azimuth_deg),
            )
            frame = renderer.render(camera)
            composite = np.clip(
                frame.premultiplied_rgb
                + (1.0 - frame.alpha) * args.background,
                0.0,
                1.0,
            )
            rgb_u8 = np.round(composite * 255.0).astype(np.uint8)
            mask = frame.alpha[..., 0] > args.alpha_threshold
            bbox = _bbox(mask)
            touches_border = bool(
                bbox is not None
                and (
                    bbox[0] == 0
                    or bbox[1] == 0
                    or bbox[2] == args.width - 1
                    or bbox[3] == args.height - 1
                )
            )
            path = frame_dir / f"orbit_{index:03d}.png"
            iio.imwrite(path, rgb_u8)
            rendered.append(rgb_u8)
            render_times_ms.append(frame.render_ms)
            records.append(
                {
                    "index": index,
                    "azimuth_deg": azimuth_deg,
                    "path": str(path.relative_to(args.output)),
                    "alpha_support_fraction": float(np.mean(mask)),
                    "alpha_bbox_xyxy": bbox,
                    "touches_border": touches_border,
                    "render_ms": frame.render_ms,
                }
            )
        memory = renderer.memory_usage()
    finally:
        renderer.close()

    video_frames = np.stack(rendered[:-1])
    iio.imwrite(
        args.output / "orbit_360.mp4",
        video_frames,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
    )
    audit_indices = [round(index * args.frames / 12) % args.frames for index in range(12)]
    iio.imwrite(
        args.output / "orbit_contact_sheet.png",
        _contact_sheet([rendered[index] for index in audit_indices]),
    )

    adjacent_mae = [
        float(np.mean(np.abs(rendered[index].astype(np.float32) - rendered[index - 1])))
        / 255.0
        for index in range(1, len(rendered))
    ]
    closure_mae = (
        float(np.mean(np.abs(rendered[-1].astype(np.float32) - rendered[0]))) / 255.0
    )
    support = [record["alpha_support_fraction"] for record in records]
    accepted_numeric = bool(
        len(rendered) == args.frames + 1
        and min(support) > 0.01
        and not any(record["touches_border"] for record in records)
        and closure_mae <= 1.0 / 255.0
    )
    report = {
        "schema_version": "radeon_oneloop.gaussian_orbit_audit.v1",
        "formal": False,
        "accepted_numeric": accepted_numeric,
        "visual_review_required": True,
        "eligible_for_heldout_real_metrics": False,
        "asset": asset_audit,
        "orbit": {
            "frames_without_duplicate_endpoint": args.frames,
            "azimuth_start_deg": 0.0,
            "azimuth_end_deg": 360.0,
            "distance_m": 0.3,
            "image_size_wh": [args.width, args.height],
            "fps": args.fps,
            "cycle_closure_rgb_mae": closure_mae,
            "adjacent_rgb_mae": {
                "mean": statistics.fmean(adjacent_mae),
                "p95": float(np.percentile(adjacent_mae, 95)),
                "max": max(adjacent_mae),
            },
            "alpha_support_fraction": {
                "min": min(support),
                "mean": statistics.fmean(support),
                "max": max(support),
            },
            "border_contact_frames": sum(record["touches_border"] for record in records),
        },
        "render_ms": {
            "mean": statistics.fmean(render_times_ms),
            "p50": float(np.percentile(render_times_ms, 50)),
            "p95": float(np.percentile(render_times_ms, 95)),
            "max": max(render_times_ms),
        },
        "vksplat_memory": memory,
        "frames": records,
        "outputs": {
            "video": "orbit_360.mp4",
            "contact_sheet": "orbit_contact_sheet.png",
        },
        "not_proven": [
            "held-out real-view quality",
            "photometric accuracy",
            "task success",
        ],
        "physical_output": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not accepted_numeric:
        raise RuntimeError("Gaussian orbit numeric gate failed")


if __name__ == "__main__":
    main()
