#!/usr/bin/env python3
"""Create a contact sheet, video, and diagnostics for a 49-frame VkSplat orbit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from gaussian.vksplat_render_ply import sha256_file


INDICES = (0, 6, 12, 18, 24, 30, 36, 42, 48)


def audit(render_root: Path, output: Path, fps: float) -> dict:
    import cv2

    root = render_root.resolve()
    manifest_path = root / "render_manifest.json"
    render_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    for index in range(49):
        path = root / f"orbit_{index:05d}.png"
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"missing orbit frame: {path}")
        frames.append(frame)
    values = np.stack(frames)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = [values[index] for index in INDICES]
    rows = [np.concatenate(selected[offset : offset + 3], axis=1) for offset in range(0, 9, 3)]
    contact = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(output / "orbit_contact.png"), contact):
        raise RuntimeError("failed to write orbit contact sheet")
    height, width = values.shape[1:3]
    writer = cv2.VideoWriter(
        str(output / "orbit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open orbit video writer")
    try:
        for frame in values:
            writer.write(frame)
    finally:
        writer.release()
    adjacent = [
        float(np.mean(np.abs(values[index].astype(np.float32) - values[index - 1])) / 255.0)
        for index in range(1, 49)
    ]
    seam = float(np.mean(np.abs(values[0].astype(np.float32) - values[-1])) / 255.0)
    metrics = {
        "schema_version": "radeon_oneloop.hybrid_vksplat_orbit_audit.v1",
        "formal": False,
        "eligible_for_heldout_real_metrics": False,
        "render_manifest_sha256": sha256_file(manifest_path),
        "frames": 49,
        "image_size_wh": [width, height],
        "fps": fps,
        "adjacent_rgb_mae": {
            "mean": float(np.mean(adjacent)),
            "p95": float(np.percentile(adjacent, 95)),
            "max": float(np.max(adjacent)),
        },
        "last_to_first_nominal_7_3469deg_rgb_mae": seam,
        "review_status": "pending_human_floater_identity_and_seam_review",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE"}:
            lines.append(f"{sha256_file(path)}  {path.name}")
    hashes = output / "hashes.sha256"
    hashes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "DONE").write_text(
        json.dumps(
            {
                "schema_version": metrics["schema_version"],
                "status": "audit_complete_pending_human_review",
                "metrics_sha256": sha256_file(output / "metrics.json"),
                "hashes_sha256": sha256_file(hashes),
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
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    print(json.dumps(audit(args.render_root, args.output, args.fps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
