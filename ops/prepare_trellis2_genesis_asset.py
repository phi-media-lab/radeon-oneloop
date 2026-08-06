#!/usr/bin/env python3
"""Canonicalize a TRELLIS.2 GLB for Genesis visual/collision separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


ASSET_STEM = "graffiti_mickey_trellis2_real_front_seed12345"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--collision-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-height-m", type=float, default=0.095)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    collision = args.collision_mesh.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not collision.is_file():
        raise FileNotFoundError(collision)
    if args.target_height_m <= 0.0:
        raise ValueError("target-height-m must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(source, force="scene", process=False)
    mesh = loaded.to_geometry()
    source_bounds = mesh.bounds.copy()
    source_vertices = len(mesh.vertices)
    source_faces = len(mesh.faces)

    # This TRELLIS.2/glTF output: +Y up, +Z front and -X viewer-left. One
    # proper rotation maps it to
    # the project contract: +Z up, +Y front, +X viewer-left.
    source_to_canonical = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh.apply_transform(source_to_canonical)
    canonical_height = float(mesh.extents[2])
    scale = args.target_height_m / canonical_height
    mesh.apply_scale(scale)
    metric_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-metric_center)

    visual_obj = output_dir / f"{ASSET_STEM}_visual.obj"
    visual_mtl = output_dir / f"{ASSET_STEM}_visual.mtl"
    visual_texture = output_dir / f"{ASSET_STEM}_texture.png"
    urdf = output_dir / f"{ASSET_STEM}.urdf"
    manifest = output_dir / f"{ASSET_STEM}.json"

    obj_text, sidecars = trimesh.exchange.obj.export_obj(
        mesh,
        include_texture=True,
        return_texture=True,
        digits=9,
    )
    material_bytes = sidecars.get("material.mtl")
    texture_bytes = sidecars.get("material_0.png")
    if material_bytes is None or texture_bytes is None:
        raise RuntimeError("GLB texture could not be exported to OBJ/MTL")
    obj_text = obj_text.replace("mtllib material.mtl", f"mtllib {visual_mtl.name}")
    material_text = material_bytes.decode("utf-8").replace(
        "material_0.png", visual_texture.name
    )
    visual_obj.write_text(obj_text, encoding="utf-8")
    visual_mtl.write_text(material_text, encoding="utf-8")
    visual_texture.write_bytes(texture_bytes)

    collision_relative = Path(collision.name)
    if collision.parent != output_dir:
        raise ValueError("collision mesh must be in output-dir for portable URDF paths")

    # The inertial is an ellipsoid approximation for a 40 g plush/vinyl toy.
    # Physics geometry remains the closed, low-complexity collision proxy; the
    # generated non-watertight mesh is never used for contact or inertia.
    mass_kg = 0.04
    collision_mesh = trimesh.load(collision, force="mesh", process=False)
    half_extents = np.asarray(collision_mesh.extents, dtype=np.float64) / 2.0
    inertia = mass_kg / 5.0 * np.array(
        [
            half_extents[1] ** 2 + half_extents[2] ** 2,
            half_extents[0] ** 2 + half_extents[2] ** 2,
            half_extents[0] ** 2 + half_extents[1] ** 2,
        ]
    )
    urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="{ASSET_STEM}">
  <link name="object">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass_kg:.9f}"/>
      <inertia ixx="{inertia[0]:.12g}" ixy="0" ixz="0" iyy="{inertia[1]:.12g}" iyz="0" izz="{inertia[2]:.12g}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{visual_obj.name}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{collision_relative}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    report = {
        "schema_version": "radeon_oneloop.trellis2_genesis_asset.v1",
        "formal": False,
        "source": {
            "path": source.name,
            "sha256": sha256_file(source),
            "coordinate_convention": {
                "front": "+Z",
                "up": "+Y",
                "viewer_left": "-X",
            },
            "bounds": source_bounds.tolist(),
            "vertices": source_vertices,
            "faces": source_faces,
            "watertight": bool(mesh.is_watertight),
        },
        "canonical_visual": {
            "mesh": visual_obj.name,
            "mesh_sha256": sha256_file(visual_obj),
            "material": visual_mtl.name,
            "material_sha256": sha256_file(visual_mtl),
            "texture": visual_texture.name,
            "texture_sha256": sha256_file(visual_texture),
            "coordinate_convention": {
                "front": "+Y",
                "up": "+Z",
                "viewer_left": "+X",
                "unit": "m",
            },
            "uniform_scale": scale,
            "center_offset_before_recentering_m": metric_center.tolist(),
            "bounds_m": mesh.bounds.tolist(),
            "extents_m": mesh.extents.tolist(),
            "visual_only": True,
        },
        "collision": {
            "mesh": collision.name,
            "sha256": sha256_file(collision),
            "extents_m": collision_mesh.extents.tolist(),
            "generated_visual_used_for_collision": False,
        },
        "urdf": {"path": urdf.name, "sha256": sha256_file(urdf)},
        "metric_anchor": {
            "dimension": "overall_height",
            "value_m": args.target_height_m,
            "status": "user_confirmed_metric_anchor",
        },
        "limitations": [
            "rear appearance remains a single-view generative hypothesis",
            "visual mesh is non-watertight and excluded from physics",
            "mass and inertia are nonformal priors pending measurement",
        ],
    }
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
