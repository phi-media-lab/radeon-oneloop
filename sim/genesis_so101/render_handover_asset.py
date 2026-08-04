#!/usr/bin/env python3
"""Render a fixed canonical front view of the handover visual on AMD Genesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .handover_asset import DEFAULT_MESH, sha256_file


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    mesh = args.mesh.resolve()
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    args.output.mkdir(parents=True, exist_ok=True)

    import genesis as gs

    gs.init(backend=gs.amdgpu, seed=args.seed)
    if gs.backend != gs.amdgpu:
        raise RuntimeError(f"Genesis did not select AMD GPU: {gs.backend}")
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            ambient_light=(0.72, 0.72, 0.72),
            shadow=True,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Mesh(
            file=str(mesh),
            pos=(0.0, 0.0, 0.045),
            fixed=True,
            collision=False,
            convexify=False,
            decimate=False,
        )
    )
    camera = scene.add_camera(
        res=(640, 640),
        pos=(0.0, 0.42, 0.085),
        lookat=(0.0, 0.0, 0.045),
        up=(0.0, 0.0, 1.0),
        fov=22,
        GUI=False,
    )
    scene.build()
    rgb, _, _, _ = camera.render(rgb=True)
    image = as_numpy(rgb).astype(np.uint8)
    image_path = args.output / "graffiti_mickey_front.png"
    iio.imwrite(image_path, image)
    report = {
        "schema_version": "radeon_oneloop.graffiti_mickey_preview.v1",
        "formal": False,
        "backend": str(gs.backend),
        "device": str(gs.device),
        "mesh": mesh.name,
        "mesh_sha256": sha256_file(mesh),
        "canonical_orientation": {"front": "+Y", "up": "+Z", "viewer_left": "+X"},
        "image": image_path.name,
        "image_sha256": sha256_file(image_path),
        "image_shape": list(image.shape),
        "purpose": "visual and material QA only; not a physics or performance metric",
    }
    metrics = args.output / "metrics.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "hashes.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in (image_path, metrics)
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
