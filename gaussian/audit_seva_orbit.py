#!/usr/bin/env python3
"""Audit a SEVA four-view orbit before it may supply generated pseudo-views."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from gaussian.prepare_four_view_generation import sha256_file, validate_generation_input
from gaussian.provenance_quarantine import assert_not_quarantined


RUN_SCHEMA = "radeon_oneloop.seva_four_view_orbit.v1"
AUDIT_SCHEMA = "radeon_oneloop.seva_four_view_orbit_audit.v1"
DONE_SCHEMA = "radeon_oneloop.seva_four_view_orbit_audit_done.v1"
CONTACT_INDICES = (0, 6, 12, 18, 24, 30, 36, 42, 48)
ANCHORS = (("front", 0), ("left", 12), ("back", 24), ("right", 37))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def mask_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=bool)
    b = np.asarray(rhs, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask IoU requires matching shapes")
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def infer_foreground(frame: np.ndarray, *, threshold: float = 0.055) -> np.ndarray:
    """Infer the largest foreground component relative to the border color."""

    import cv2

    value = np.asarray(frame, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("foreground inference requires an H x W x 3 frame")
    border_width = max(2, min(value.shape[:2]) // 48)
    border = np.concatenate(
        (
            value[:border_width].reshape(-1, 3),
            value[-border_width:].reshape(-1, 3),
            value[:, :border_width].reshape(-1, 3),
            value[:, -border_width:].reshape(-1, 3),
        )
    )
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.mean(np.abs(value.astype(np.float32) - background), axis=2) / 255.0
    raw = (distance >= threshold).astype(np.uint8)
    kernel = np.ones((7, 7), dtype=np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if count <= 1:
        return np.zeros(value.shape[:2], dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled > 0


def masked_rgb_mae(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray) -> float:
    support = np.asarray(mask, dtype=bool)
    if not np.any(support):
        raise ValueError("masked RGB MAE requires nonempty support")
    difference = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32)) / 255.0
    return float(difference[support].mean())


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("foreground mask is empty")
    height, width = mask.shape
    return float(xs.mean() / width), float(ys.mean() / height)


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _read_observed_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.ndim != 3 or value.shape[2] != 4:
        raise ValueError(f"observed anchor must be RGBA: {path}")
    return cv2.cvtColor(value[:, :, :3], cv2.COLOR_BGR2RGB), value[:, :, 3] > 127


def _write_rgb(path: Path, value: np.ndarray) -> None:
    import cv2

    if not cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write image: {path}")


def _contact_sheet(frames: np.ndarray) -> np.ndarray:
    selected = [frames[index] for index in CONTACT_INDICES]
    rows = [np.concatenate(selected[offset : offset + 3], axis=1) for offset in range(0, 9, 3)]
    return np.concatenate(rows, axis=0)


def _anchor_comparison(
    frames: np.ndarray, observed: dict[str, tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    real_row = np.concatenate([observed[label][0] for label, _ in ANCHORS], axis=1)
    generated_row = np.concatenate([frames[index] for _, index in ANCHORS], axis=1)
    difference_row = np.concatenate(
        [
            np.clip(
                np.abs(frames[index].astype(np.int16) - observed[label][0].astype(np.int16))
                * 3,
                0,
                255,
            ).astype(np.uint8)
            for label, index in ANCHORS
        ],
        axis=1,
    )
    return np.concatenate((real_row, generated_row, difference_row), axis=0)


def _validate_run(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"
    done_path = run_root / "DONE"
    if not manifest_path.is_file() or not done_path.is_file():
        raise ValueError("SEVA run requires manifest.json and DONE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unexpected SEVA run schema")
    if manifest.get("formal") is not False or manifest.get("review_status") != (
        "pending_identity_temporal_and_loop_audit"
    ):
        raise ValueError("SEVA run provenance or review boundary was weakened")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("SEVA DONE does not bind its manifest")
    if done.get("hashes_sha256") != sha256_file(run_root / "hashes.sha256"):
        raise ValueError("SEVA DONE does not bind its hash index")
    assert_not_quarantined([("seva_manifest", manifest)])
    return manifest


def audit(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    run_root = args.run_root.resolve()
    input_root = args.input_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    run = _validate_run(run_root)
    generation_input = validate_generation_input(input_root)
    if run["input"]["four_view_manifest_sha256"] != sha256_file(input_root / "manifest.json"):
        raise ValueError("SEVA run and four-view input do not match")

    frame_paths = [run_root / record["relpath"] for record in run["frames"]]
    if len(frame_paths) != 49 or any(not path.is_file() for path in frame_paths):
        raise ValueError("SEVA run does not contain 49 bound frames")
    frames = np.stack([_read_rgb(path) for path in frame_paths])
    if frames.shape != (49, 576, 576, 3):
        raise ValueError(f"unexpected SEVA frame tensor: {frames.shape}")

    observed_records = {
        item["generator_label"]: item for item in generation_input["observed_inputs"]
    }
    observed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, _ in ANCHORS:
        record = observed_records[label]
        rgb, mask = _read_observed_rgba(input_root / record["hunyuan_rgba_relpath"])
        if rgb.shape != (576, 576, 3) or mask.shape != (576, 576):
            raise ValueError("observed anchors differ from the 576p SEVA contract")
        observed[label] = (rgb, mask)

    generated_masks = np.stack([infer_foreground(frame) for frame in frames])
    if any(not np.any(mask) for mask in generated_masks):
        raise ValueError("one or more generated frames has no inferred foreground")

    anchor_iou: dict[str, float] = {}
    anchor_rgb: dict[str, float] = {}
    for label, index in ANCHORS:
        real_rgb, real_mask = observed[label]
        anchor_iou[label] = mask_iou(generated_masks[index], real_mask)
        anchor_rgb[label] = masked_rgb_mae(frames[index], real_rgb, real_mask)

    adjacent_rgb = [
        float(np.abs(frames[index].astype(np.float32) - frames[index - 1]).mean() / 255.0)
        for index in range(1, 49)
    ]
    adjacent_mask_iou = [
        mask_iou(generated_masks[index], generated_masks[index - 1]) for index in range(1, 49)
    ]
    seam_rgb = float(np.abs(frames[0].astype(np.float32) - frames[-1]).mean() / 255.0)
    seam_mask_iou = mask_iou(generated_masks[0], generated_masks[-1])
    areas = np.asarray([np.mean(mask) for mask in generated_masks], dtype=np.float64)
    centers = np.asarray([centroid(mask) for mask in generated_masks], dtype=np.float64)

    output.mkdir(parents=True)
    mask_root = output / "generated_masks"
    mask_root.mkdir()
    for index, mask in enumerate(generated_masks):
        if not cv2.imwrite(str(mask_root / f"{index:05d}.png"), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write generated mask {index}")
    _write_rgb(output / "generated_contact.png", _contact_sheet(frames))
    _write_rgb(output / "real_generated_difference_anchors.png", _anchor_comparison(frames, observed))

    metrics = {
        "schema_version": AUDIT_SCHEMA,
        "created_utc": utc_now(),
        "formal": False,
        "eligible_for_formal_metrics": False,
        "eligible_for_heldout_real_metrics": False,
        "seva_manifest_sha256": sha256_file(run_root / "manifest.json"),
        "four_view_manifest_sha256": sha256_file(input_root / "manifest.json"),
        "frames": 49,
        "image_size_wh": [576, 576],
        "anchor_schedule": {label: index for label, index in ANCHORS},
        "real_anchor_silhouette_iou": {
            "values": anchor_iou,
            "mean": statistics.fmean(anchor_iou.values()),
            "min": min(anchor_iou.values()),
        },
        "real_anchor_rgb_mae_on_real_support": {
            "values": anchor_rgb,
            "mean": statistics.fmean(anchor_rgb.values()),
            "max": max(anchor_rgb.values()),
        },
        "adjacent_rgb_mae": {
            "mean": statistics.fmean(adjacent_rgb),
            "p95": float(np.percentile(adjacent_rgb, 95)),
            "max": max(adjacent_rgb),
        },
        "adjacent_foreground_iou": {
            "mean": statistics.fmean(adjacent_mask_iou),
            "p05": float(np.percentile(adjacent_mask_iou, 5)),
            "min": min(adjacent_mask_iou),
        },
        "cyclic_seam": {
            "first_last_rgb_mae": seam_rgb,
            "first_last_foreground_iou": seam_mask_iou,
            "rgb_mae_over_adjacent_p95": seam_rgb / max(float(np.percentile(adjacent_rgb, 95)), 1e-12),
        },
        "foreground_stability": {
            "area_fraction_mean": float(areas.mean()),
            "area_fraction_cv": float(areas.std() / max(areas.mean(), 1e-12)),
            "centroid_x_range_normalized": float(np.ptp(centers[:, 0])),
            "centroid_y_range_normalized": float(np.ptp(centers[:, 1])),
        },
        "generated_mask": {
            "method": "largest_filled_component_relative_to_border_median",
            "threshold": 0.055,
            "relpath": "generated_masks",
        },
        "review_status": "pending_human_identity_topology_and_loop_review",
        "metric_boundary": [
            "Generated views are never observed or held-out-real evidence.",
            "Anchor metrics measure preservation of the four inputs, not hidden-side correctness.",
            "Only a human identity/topology review may promote these frames to low-confidence pseudo-views.",
        ],
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hash_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE"}:
            hash_lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    hashes_path = output / "hashes.sha256"
    hashes_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    (output / "DONE").write_text(
        json.dumps(
            {
                "schema_version": DONE_SCHEMA,
                "status": "audit_complete_pending_human_review",
                "metrics_sha256": sha256_file(metrics_path),
                "hashes_sha256": sha256_file(hashes_path),
                "completed_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(audit(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
