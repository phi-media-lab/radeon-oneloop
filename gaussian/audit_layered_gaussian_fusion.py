#!/usr/bin/env python3
"""Compare a layered Gaussian preview against its unchanged observed core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from gaussian.vksplat_render_ply import sha256_file


SCHEMA = "radeon_oneloop.layered_gaussian_fusion_audit.v1"
FRAME_COUNT = 49
ANCHORS = (0, 12, 24, 37)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_frames(root: Path) -> np.ndarray:
    import cv2

    frames = []
    for index in range(FRAME_COUNT):
        path = root / f"orbit_{index:05d}.png"
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"missing orbit frame: {path}")
        frames.append(frame.astype(np.float32) / 255.0)
    values = np.stack(frames)
    if values.shape[1:] != frames[0].shape:
        raise ValueError("render frames have inconsistent dimensions")
    return values


def audit(observed_root: Path, fused_root: Path, output: Path) -> dict:
    observed_root = observed_root.resolve()
    fused_root = fused_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    observed = _load_frames(observed_root)
    fused = _load_frames(fused_root)
    if observed.shape != fused.shape:
        raise ValueError("observed and fused renders have different shapes")

    frame_metrics = []
    for index, (source, candidate) in enumerate(zip(observed, fused, strict=True)):
        source_mask = np.any(source < 0.98, axis=2)
        candidate_mask = np.any(candidate < 0.98, axis=2)
        intersection = int(np.count_nonzero(source_mask & candidate_mask))
        union = int(np.count_nonzero(source_mask | candidate_mask))
        frame_metrics.append(
            {
                "index": index,
                "rgb_mae": float(np.mean(np.abs(source - candidate))),
                "foreground_mask_iou": intersection / union if union else 1.0,
                "new_foreground_fraction": float(
                    np.count_nonzero(candidate_mask & ~source_mask) / source_mask.size
                ),
                "lost_foreground_fraction": float(
                    np.count_nonzero(source_mask & ~candidate_mask) / source_mask.size
                ),
            }
        )

    anchor_metrics = [frame_metrics[index] for index in ANCHORS]
    anchor_rgb_mae_max = max(value["rgb_mae"] for value in anchor_metrics)
    anchor_mask_iou_min = min(value["foreground_mask_iou"] for value in anchor_metrics)
    safety_pass = anchor_rgb_mae_max <= 0.005 and anchor_mask_iou_min >= 0.98
    metrics = {
        "schema_version": SCHEMA,
        "created_utc": _utc_now(),
        "formal": False,
        "eligible_for_formal_metrics": False,
        "eligible_for_heldout_real_metrics": False,
        "observed_render_manifest_sha256": sha256_file(observed_root / "render_manifest.json"),
        "fused_render_manifest_sha256": sha256_file(fused_root / "render_manifest.json"),
        "frame_count": FRAME_COUNT,
        "anchor_indices": list(ANCHORS),
        "anchor_metrics": anchor_metrics,
        "summary": {
            "mean_rgb_mae": float(np.mean([value["rgb_mae"] for value in frame_metrics])),
            "max_rgb_mae": max(value["rgb_mae"] for value in frame_metrics),
            "mean_foreground_mask_iou": float(
                np.mean([value["foreground_mask_iou"] for value in frame_metrics])
            ),
            "anchor_rgb_mae_max": anchor_rgb_mae_max,
            "anchor_foreground_mask_iou_min": anchor_mask_iou_min,
        },
        "gates": {
            "observed_anchor_safety": {
                "status": "pass" if safety_pass else "fail",
                "rgb_mae_max_threshold": 0.005,
                "foreground_mask_iou_min_threshold": 0.98,
            },
            "generated_fill_effectiveness": {
                "status": "inconclusive_without_heldout_real_views",
                "reason": "novel-view plausibility is not ground-truth completion accuracy",
            },
        },
        "decision": (
            "accept_as_optional_nonformal_toggle_default_off"
            if safety_pass
            else "reject_layered_preview"
        ),
    }
    output.mkdir(parents=True)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "DONE").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "audit_complete",
                "metrics_sha256": sha256_file(metrics_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-render-root", type=Path, required=True)
    parser.add_argument("--fused-render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.observed_render_root, args.fused_render_root, args.output), indent=2))


if __name__ == "__main__":
    main()
