#!/usr/bin/env python3
"""Render one accepted canonical camera through the runtime appearance layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .gaussian_appearance import (
    PinholeCamera,
    VKSPLAT_COMMIT,
    VkSplatAppearanceRenderer,
    observed_core_asset,
    probe_nyx,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    if args.camera_index not in range(4):
        raise ValueError("camera-index must be between zero and three")
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v3 as iio

    asset = observed_core_asset(args.asset_root)
    asset_audit = asset.validate()
    cameras_document = json.loads(asset.cameras_path.read_text(encoding="utf-8"))
    camera_record = cameras_document["cameras"][args.camera_index]
    width, height = (int(value) for value in camera_record["image_size_wh"])
    camera = PinholeCamera(
        width=width,
        height=height,
        intrinsic_3x3=np.asarray(camera_record["intrinsic_3x3"]),
        camera_from_object_opencv_4x4=np.asarray(
            camera_record["world_to_camera_opencv_4x4"]
        ),
    )
    nyx = probe_nyx()
    renderer = VkSplatAppearanceRenderer(asset, args.vksplat_root)
    try:
        frame = renderer.render(camera)
        memory = renderer.memory_usage()
    finally:
        renderer.close()

    background = 0.125
    composite = np.clip(
        frame.premultiplied_rgb + (1.0 - frame.alpha) * background,
        0.0,
        1.0,
    )
    image_path = args.output / "canonical_probe.png"
    iio.imwrite(image_path, np.round(composite * 255.0).astype(np.uint8))
    report = {
        "schema_version": "radeon_oneloop.gaussian_appearance_probe.v1",
        "formal": False,
        "asset": asset_audit,
        "camera": {
            "view_id": camera_record["view_id"],
            "image_size_wh": [width, height],
            "model": "PINHOLE_OPENCV",
        },
        "coordinate_contract": {
            "genesis_camera": "T_world_camera_opengl",
            "object": "T_world_object_canonical",
            "vksplat": "T_camera_object_opencv",
            "formula": "inverse(T_world_camera_opengl * diag(1,-1,-1,1)) * T_world_object_canonical",
        },
        "nyx": {
            "available": nyx.available,
            "backend": nyx.backend,
            "reason": nyx.reason,
            "details": nyx.details,
        },
        "vksplat": {
            "available": True,
            "commit": VKSPLAT_COMMIT,
            "root": str(args.vksplat_root.resolve()),
            "render_ms": frame.render_ms,
            "mean_alpha": float(frame.alpha.mean()),
            "max_alpha": float(frame.alpha.max()),
            "nonzero_alpha_fraction": float(np.mean(frame.alpha[..., 0] > 1.0e-3)),
            "memory": memory,
        },
        "fallback": "genesis_debug_mesh",
        "output": image_path.name,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
