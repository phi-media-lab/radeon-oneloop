#!/usr/bin/env python3
"""Select a Gaussian alpha threshold against an accepted carrier silhouette."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from gaussian.prepare_vista4d_object_input import (
    VISTA4D_FRAMES,
    load_surface_carrier_source,
    vista4d_camera_track,
)
from sim.genesis_so101.gaussian_appearance import (
    PinholeCamera,
    VkSplatAppearanceRenderer,
    nonformal_candidate_asset,
)
from sim.genesis_so101.gaussian_orbit_audit import (
    canonical_orbit_extrinsic,
    scaled_intrinsic,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_mask_alignment(source: np.ndarray, point: np.ndarray) -> dict[str, float]:
    source_mask = np.asarray(source, dtype=bool)
    point_mask = np.asarray(point, dtype=bool)
    if source_mask.shape != point_mask.shape:
        raise ValueError("mask alignment requires matching shapes")
    union = source_mask | point_mask
    union_count = int(np.count_nonzero(union))
    if not union_count:
        return {
            "iou": 1.0,
            "source_support_fraction": 0.0,
            "point_support_fraction": 0.0,
            "source_only_fraction_of_union": 0.0,
            "point_only_fraction_of_union": 0.0,
        }
    return {
        "iou": float(np.count_nonzero(source_mask & point_mask) / union_count),
        "source_support_fraction": float(np.mean(source_mask)),
        "point_support_fraction": float(np.mean(point_mask)),
        "source_only_fraction_of_union": float(
            np.count_nonzero(source_mask & ~point_mask) / union_count
        ),
        "point_only_fraction_of_union": float(
            np.count_nonzero(point_mask & ~source_mask) / union_count
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--surface-carrier-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--distance-m", type=float, default=0.3)
    parser.add_argument(
        "--thresholds",
        default="0.001,0.005,0.01,0.025,0.05,0.1,0.2,0.35,0.5",
    )
    args = parser.parse_args()
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("alpha thresholds must be in (0, 1)")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("alpha thresholds must be unique")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    asset = nonformal_candidate_asset(args.asset_root)
    asset_audit = asset.validate()
    cameras_document = json.loads(asset.cameras_path.read_text(encoding="utf-8"))
    front = cameras_document["cameras"][0]
    source_size = tuple(int(value) for value in front["image_size_wh"])
    output_size = (args.width, args.height)
    intrinsic = scaled_intrinsic(
        np.asarray(front["intrinsic_3x3"], dtype=np.float64),
        source_size,
        output_size,
    )
    target_c2w, target_intrinsics = vista4d_camera_track(
        frames=VISTA4D_FRAMES,
        intrinsic_3x3=intrinsic,
        distance_m=args.distance_m,
    )
    _, carrier_alpha, carrier_record = load_surface_carrier_source(
        args.surface_carrier_root,
        width=args.width,
        height=args.height,
        target_c2w=target_c2w,
        target_intrinsics=target_intrinsics,
    )

    records: dict[float, list[dict[str, float]]] = {value: [] for value in thresholds}
    render_ms = []
    renderer = VkSplatAppearanceRenderer(asset, args.vksplat_root)
    try:
        for index in range(VISTA4D_FRAMES):
            camera = PinholeCamera(
                width=args.width,
                height=args.height,
                intrinsic_3x3=intrinsic,
                camera_from_object_opencv_4x4=canonical_orbit_extrinsic(
                    360.0 * index / VISTA4D_FRAMES, distance_m=args.distance_m
                ),
            )
            rendered = renderer.render(camera)
            render_ms.append(rendered.render_ms)
            source_mask = carrier_alpha[index] >= 0.5
            gaussian_alpha = rendered.alpha[..., 0]
            for threshold in thresholds:
                records[threshold].append(
                    binary_mask_alignment(source_mask, gaussian_alpha >= threshold)
                )
    finally:
        renderer.close()

    candidates = []
    for threshold in thresholds:
        values = records[threshold]
        candidates.append(
            {
                "alpha_threshold": threshold,
                "silhouette_iou_mean": statistics.fmean(value["iou"] for value in values),
                "silhouette_iou_p05": float(
                    np.percentile([value["iou"] for value in values], 5)
                ),
                "source_support_fraction_mean": statistics.fmean(
                    value["source_support_fraction"] for value in values
                ),
                "point_support_fraction_mean": statistics.fmean(
                    value["point_support_fraction"] for value in values
                ),
                "source_only_fraction_of_union_mean": statistics.fmean(
                    value["source_only_fraction_of_union"] for value in values
                ),
                "point_only_fraction_of_union_mean": statistics.fmean(
                    value["point_only_fraction_of_union"] for value in values
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda value: (
            value["silhouette_iou_mean"],
            value["silhouette_iou_p05"],
            -value["alpha_threshold"],
        ),
    )
    report = {
        "schema_version": "radeon_oneloop.vista4d_mask_alignment_audit.v1",
        "formal": False,
        "physical_output": False,
        "eligible_for_heldout_real_metrics": False,
        "frames": VISTA4D_FRAMES,
        "image_size_wh": list(output_size),
        "asset": asset_audit,
        "surface_carrier": carrier_record,
        "candidates": candidates,
        "selected": selected,
        "selection_rule": "maximum mean IoU, then p05 IoU, then lower threshold",
        "render_ms": {
            "mean": statistics.fmean(render_ms),
            "p95": float(np.percentile(render_ms, 95)),
            "max": max(render_ms),
        },
        "allowed_role": "conditioning_mask_threshold_selection_only",
        "not_proven": [
            "Vista4D output quality",
            "hidden geometry correctness",
            "held-out real-view quality",
        ],
    }
    report_path = args.output / "metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "DONE").write_text(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "status": "complete_nonformal_threshold_selection",
                "metrics_sha256": sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
