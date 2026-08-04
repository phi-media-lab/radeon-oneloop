#!/usr/bin/env python3
"""Create a provenance-bound visual and numeric audit of a Vista4D proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONTACT_INDICES = (0, 6, 12, 18, 24, 30, 36, 42, 48)
ANCHOR_INDICES = (0, 12, 24, 37)
AUDIT_SCHEMA = "radeon_oneloop.vista4d_completion_visual_audit.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_video(path: Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def read_masks(path: Path) -> np.ndarray:
    import cv2

    paths = sorted(path.glob("*.png"))
    masks = []
    for item in paths:
        mask = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask: {item}")
        masks.append(mask > 127)
    if not masks:
        raise RuntimeError(f"no masks found in {path}")
    return np.stack(masks)


def masked_mae(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        raise ValueError("masked MAE requires nonempty support")
    difference = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32)) / 255.0
    return float(difference[mask].mean())


def mask_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=bool)
    b = np.asarray(rhs, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask IoU requires matching shapes")
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def inferred_foreground(frame: np.ndarray, *, threshold: float = 0.055) -> np.ndarray:
    """Infer the main object against Vista4D's nearly uniform border background."""

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


def make_contact_sheet(frames: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    selected = [frames[index] for index in indices]
    rows = [np.concatenate(selected[offset : offset + 3], axis=1) for offset in range(0, 9, 3)]
    return np.concatenate(rows, axis=0)


def make_anchor_comparison(
    source: np.ndarray, generated: np.ndarray, indices: tuple[int, ...]
) -> np.ndarray:
    source_row = np.concatenate([source[index] for index in indices], axis=1)
    generated_row = np.concatenate([generated[index] for index in indices], axis=1)
    difference_row = np.concatenate(
        [
            np.clip(
                np.abs(generated[index].astype(np.int16) - source[index].astype(np.int16))
                * 3,
                0,
                255,
            ).astype(np.uint8)
            for index in indices
        ],
        axis=1,
    )
    return np.concatenate((source_row, generated_row, difference_row), axis=0)


def write_binary_masks(root: Path, masks: np.ndarray) -> None:
    """Persist inferred masks so downstream pseudo-views use the audited support."""

    import cv2

    values = np.asarray(masks, dtype=bool)
    if values.ndim != 3 or values.shape[0] != 49:
        raise ValueError("generated mask bundle must contain 49 H x W masks")
    root.mkdir(parents=True, exist_ok=False)
    for index, mask in enumerate(values):
        if not cv2.imwrite(str(root / f"{index:05d}.png"), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write generated mask {index}")


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--proposal-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    conditioning = args.conditioning.resolve()
    proposal_run = args.proposal_run.resolve()
    proposal = json.loads((proposal_run / "manifest.json").read_text(encoding="utf-8"))
    done = json.loads((proposal_run / "DONE").read_text(encoding="utf-8"))
    if proposal.get("schema_version") != "radeon_oneloop.vista4d_object_completion_proposal.v1":
        raise RuntimeError("unexpected Vista4D proposal schema")
    if done.get("status") != "done_candidate_pending_visual_review":
        raise RuntimeError("Vista4D proposal is incomplete")
    if proposal.get("formal") is not False or proposal.get("physical_output") is not False:
        raise RuntimeError("proposal provenance boundary was weakened")
    generated_path = proposal_run / "inference" / f"video_seed={proposal['model']['seed']}.mp4"
    if sha256_file(generated_path) != proposal["generated_video_sha256"]:
        raise RuntimeError("generated video does not match its proposal manifest")

    generated = read_video(generated_path)
    point_cloud = read_video(conditioning / "video_pc.mp4")
    source = read_video(conditioning / "video_src.mp4")
    masks = read_masks(conditioning / "alpha_mask_pc")
    source_masks = read_masks(conditioning / "alpha_mask_src")
    if generated.shape != point_cloud.shape or source.shape != point_cloud.shape:
        raise RuntimeError("proposal and conditioning videos have different shapes")
    if (
        len(generated) != 49
        or masks.shape != generated.shape[:3]
        or source_masks.shape != generated.shape[:3]
    ):
        raise RuntimeError("audit requires the 49-frame 384p conditioning contract")

    kernel = np.ones((9, 9), dtype=np.uint8)
    conditioning_union = masks | source_masks
    dilated = np.stack(
        [
            cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
            for mask in conditioning_union
        ]
    )
    generated_masks = np.stack([inferred_foreground(frame) for frame in generated])
    observed_mae = [masked_mae(generated[i], point_cloud[i], masks[i]) for i in range(49)]
    source_mae = [
        masked_mae(generated[i], source[i], source_masks[i]) for i in range(49)
    ]
    background_white_mae = []
    for index in range(49):
        background = ~dilated[index]
        white = np.full_like(generated[index], 255)
        background_white_mae.append(masked_mae(generated[index], white, background))

    temporal_residual = []
    source_temporal_residual = []
    for index in range(1, 49):
        support = conditioning_union[index] | conditioning_union[index - 1]
        generated_delta = generated[index].astype(np.float32) - generated[index - 1].astype(np.float32)
        point_delta = point_cloud[index].astype(np.float32) - point_cloud[index - 1].astype(np.float32)
        source_delta = source[index].astype(np.float32) - source[index - 1].astype(np.float32)
        temporal_residual.append(float((np.abs(generated_delta - point_delta) / 255.0)[support].mean()))
        source_temporal_residual.append(
            float((np.abs(generated_delta - source_delta) / 255.0)[support].mean())
        )

    source_mask_iou = [
        mask_iou(generated_masks[index], source_masks[index]) for index in range(49)
    ]
    point_mask_iou = [
        mask_iou(generated_masks[index], masks[index]) for index in range(49)
    ]
    source_only_recall = []
    source_only_support = []
    foreground_outside_union = []
    for index in range(49):
        source_only = source_masks[index] & ~masks[index]
        source_only_support.append(int(np.count_nonzero(source_only)))
        source_only_recall.append(
            float(np.mean(generated_masks[index][source_only]))
            if np.any(source_only)
            else 1.0
        )
        foreground = generated_masks[index]
        foreground_outside_union.append(
            float(np.mean((~dilated[index])[foreground])) if np.any(foreground) else 0.0
        )

    output = args.output.resolve()
    output.mkdir(parents=True)
    generated_mask_dir = output / "generated_masks"
    write_binary_masks(generated_mask_dir, generated_masks)
    generated_contact = make_contact_sheet(generated, CONTACT_INDICES)
    point_contact = make_contact_sheet(point_cloud, CONTACT_INDICES)
    difference = np.clip(
        np.abs(generated.astype(np.int16) - point_cloud.astype(np.int16)) * 3,
        0,
        255,
    ).astype(np.uint8)
    difference_contact = make_contact_sheet(difference, CONTACT_INDICES)
    comparisons = np.concatenate((point_contact, generated_contact, difference_contact), axis=0)
    anchor_comparison = make_anchor_comparison(source, generated, ANCHOR_INDICES)
    cv2.imwrite(str(output / "generated_contact.png"), cv2.cvtColor(generated_contact, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "pc_generated_difference_contact.png"), cv2.cvtColor(comparisons, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(output / "anchor_source_generated_difference.png"),
        cv2.cvtColor(anchor_comparison, cv2.COLOR_RGB2BGR),
    )

    metrics = {
        "schema_version": AUDIT_SCHEMA,
        "formal": False,
        "eligible_for_formal_metrics": False,
        "eligible_for_heldout_real_metrics": False,
        "physical_output": False,
        "proposal_manifest_sha256": sha256_file(proposal_run / "manifest.json"),
        "generated_video_sha256": sha256_file(generated_path),
        "conditioning_manifest_sha256": sha256_file(conditioning / "input_manifest.json"),
        "frames": 49,
        "image_size_wh": [int(generated.shape[2]), int(generated.shape[1])],
        "observed_support_rgb_mae": {
            "mean": statistics.fmean(observed_mae),
            "p95": float(np.percentile(observed_mae, 95)),
            "anchor_values": {str(index): observed_mae[index] for index in ANCHOR_INDICES},
        },
        "source_support_rgb_mae": {
            "mean": statistics.fmean(source_mae),
            "p95": float(np.percentile(source_mae, 95)),
            "anchor_values": {str(index): source_mae[index] for index in ANCHOR_INDICES},
        },
        "background_white_mae_outside_dilated_support": {
            "mean": statistics.fmean(background_white_mae),
            "p95": float(np.percentile(background_white_mae, 95)),
        },
        "temporal_delta_residual_vs_conditioning": {
            "mean": statistics.fmean(temporal_residual),
            "p95": float(np.percentile(temporal_residual, 95)),
        },
        "temporal_delta_residual_vs_source_carrier": {
            "mean": statistics.fmean(source_temporal_residual),
            "p95": float(np.percentile(source_temporal_residual, 95)),
        },
        "generated_foreground_inference": {
            "method": "largest_filled_component_relative_to_border_median",
            "color_distance_threshold": 0.055,
            "mask_bundle_relpath": "generated_masks",
            "mask_count": int(len(generated_masks)),
        },
        "generated_vs_source_silhouette_iou": {
            "mean": statistics.fmean(source_mask_iou),
            "p05": float(np.percentile(source_mask_iou, 5)),
            "min": min(source_mask_iou),
            "anchor_values": {
                str(index): source_mask_iou[index] for index in ANCHOR_INDICES
            },
        },
        "generated_vs_observed_point_silhouette_iou": {
            "mean": statistics.fmean(point_mask_iou),
            "p05": float(np.percentile(point_mask_iou, 5)),
            "min": min(point_mask_iou),
        },
        "source_only_region_recall": {
            "mean": statistics.fmean(source_only_recall),
            "p05": float(np.percentile(source_only_recall, 5)),
            "source_only_pixels_total": sum(source_only_support),
        },
        "generated_foreground_outside_dilated_conditioning_union": {
            "mean_fraction_of_generated_foreground": statistics.fmean(
                foreground_outside_union
            ),
            "p95_fraction_of_generated_foreground": float(
                np.percentile(foreground_outside_union, 95)
            ),
        },
        "closed_loop_first_last_rgb_mae": float(
            np.abs(generated[0].astype(np.float32) - generated[-1].astype(np.float32)).mean() / 255.0
        ),
        "review_status": "pending_human_identity_hole_and_temporal_screen",
        "metric_boundary": [
            "No metric in this audit proves hidden-side correctness.",
            "Observed-support MAE measures preservation, not generative improvement.",
            "Generated silhouette is an inferred diagnostic because Vista4D emits no alpha.",
            "Generated frames are excluded from held-out-real evaluation.",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hash_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE"}:
            hash_lines.append(
                f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
            )
    hashes_path = output / "hashes.sha256"
    hashes_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    done_value = {
        "schema_version": metrics["schema_version"],
        "status": "audit_complete_pending_human_review",
        "metrics_sha256": sha256_file(metrics_path),
        "hashes_sha256": sha256_file(hashes_path),
    }
    (output / "DONE").write_text(json.dumps(done_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
