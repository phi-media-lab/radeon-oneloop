"""Traceable procedural asset for the Graffiti Mickey handover target."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/handover_object.json"
DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / "assets_generated"
ASSET_STEM = "miniso_disney_fun_crash_graffiti_mickey_v1"
DEFAULT_MESH = DEFAULT_ASSET_DIR / f"{ASSET_STEM}_sim_visual.obj"
DEFAULT_COLLISION_MESH = DEFAULT_ASSET_DIR / f"{ASSET_STEM}_collision.obj"
DEFAULT_SOFT_MESH = DEFAULT_COLLISION_MESH
DEFAULT_DISPLAY_MESH = DEFAULT_ASSET_DIR / f"{ASSET_STEM}_display.obj"
DEFAULT_MATERIALS = DEFAULT_ASSET_DIR / f"{ASSET_STEM}.mtl"
DEFAULT_ASSET_MANIFEST = DEFAULT_ASSET_DIR / f"{ASSET_STEM}.json"


MATERIALS: dict[str, tuple[float, float, float]] = {
    "plush_white": (0.94, 0.94, 0.90),
    "vinyl_black": (0.025, 0.025, 0.025),
    "face_pink": (0.91, 0.64, 0.66),
    "eye_gray": (0.73, 0.74, 0.72),
    "ear_blue": (0.08, 0.66, 0.82),
    "graffiti_pink": (0.93, 0.27, 0.55),
    "shoe_yellow": (0.97, 0.72, 0.04),
    "sprinkle_cyan": (0.05, 0.72, 0.88),
    "sprinkle_yellow": (0.94, 0.81, 0.13),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HandoverObjectSpec:
    asset_name: str
    exact_sku_status: str
    semi_axes_m: tuple[float, float, float]
    nominal_overall_height_m: float
    overall_height_uncertainty_m: float
    irregularity_fraction: float
    latitude_segments: int
    longitude_segments: int
    nominal_mass_kg: float
    static_friction: float
    kinetic_friction: float
    restitution: float
    pbd: dict[str, float]
    config_sha256: str
    reference_manifest_sha256: str

    @property
    def analytic_volume_m3(self) -> float:
        """Analytic volume of the plush-body collision proxy."""

        a, b, c = self.semi_axes_m
        return 4.0 * math.pi * a * b * c / 3.0

    @property
    def rigid_density_kg_m3(self) -> float:
        return self.nominal_mass_kg / self.analytic_volume_m3


@dataclass(frozen=True)
class MeshPart:
    name: str
    material: str
    vertices: np.ndarray
    faces: np.ndarray


def load_spec(path: Path = DEFAULT_CONFIG) -> HandoverObjectSpec:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "radeon_oneloop.handover_object.v2":
        raise ValueError("unsupported handover object schema")
    geometry = payload["geometry"]
    axes = tuple(float(value) for value in geometry["soft_body_semi_axes_m"])
    if len(axes) != 3 or any(not 0.015 <= value <= 0.15 for value in axes):
        raise ValueError("soft_body_semi_axes_m must contain three plausible values")
    irregularity = float(geometry["surface_irregularity_fraction"])
    if not 0.0 <= irregularity <= 0.08:
        raise ValueError("surface irregularity must be in [0, 0.08]")
    latitudes = int(geometry["latitude_segments"])
    longitudes = int(geometry["longitude_segments"])
    if latitudes < 8 or longitudes < 16 or longitudes % 2:
        raise ValueError("mesh segment counts are too small or longitude is odd")
    evidence = payload["evidence"]
    overall_height = float(evidence["nominal_overall_height_m"])
    if not 0.05 <= overall_height <= 0.20:
        raise ValueError("nominal overall height is implausible")
    physical = payload["physical_prior"]
    mass = float(physical["nominal_mass_kg"])
    if not 0.005 <= mass <= 0.5:
        raise ValueError("nominal mass is outside the task safety envelope")
    pbd = {key: float(value) for key, value in payload["pbd_soft_proxy"].items()}
    return HandoverObjectSpec(
        asset_name=str(payload["identity"]["asset_name"]),
        exact_sku_status=str(payload["identity"]["exact_sku_status"]),
        semi_axes_m=axes,
        nominal_overall_height_m=overall_height,
        overall_height_uncertainty_m=float(evidence["overall_height_uncertainty_m"]),
        irregularity_fraction=irregularity,
        latitude_segments=latitudes,
        longitude_segments=longitudes,
        nominal_mass_kg=mass,
        static_friction=float(physical["static_friction"]),
        kinetic_friction=float(physical["kinetic_friction"]),
        restitution=float(physical["restitution"]),
        pbd=pbd,
        config_sha256=sha256_file(path),
        reference_manifest_sha256=str(evidence["reference_manifest_sha256"]),
    )


def _rotation_matrix(euler_deg: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.radians(np.asarray(euler_deg, dtype=np.float64))
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _ellipsoid_mesh(
    semi_axes: tuple[float, float, float],
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    latitude_segments: int = 16,
    longitude_segments: int = 32,
    irregularity_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = semi_axes
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, c)]
    for latitude in range(1, latitude_segments):
        theta = math.pi * latitude / latitude_segments
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for longitude in range(longitude_segments):
            phi = 2.0 * math.pi * longitude / longitude_segments
            fold = 1.0 + irregularity_fraction * (
                0.55 * math.sin(3.0 * theta + 0.4) * math.cos(5.0 * phi - 0.2)
                + 0.30 * math.sin(7.0 * phi + theta)
                + 0.15 * math.cos(4.0 * theta - 3.0 * phi)
            )
            vertices.append(
                (
                    a * fold * sin_theta * math.cos(phi),
                    b * fold * sin_theta * math.sin(phi),
                    c * fold * cos_theta,
                )
            )
    bottom = len(vertices)
    vertices.append((0.0, 0.0, -c))
    faces: list[tuple[int, int, int]] = []
    for longitude in range(longitude_segments):
        faces.append((0, 1 + longitude, 1 + (longitude + 1) % longitude_segments))
    for ring in range(latitude_segments - 2):
        current = 1 + ring * longitude_segments
        following = current + longitude_segments
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            a0 = current + longitude
            a1 = current + next_longitude
            b0 = following + longitude
            b1 = following + next_longitude
            faces.append((a0, b0, b1))
            faces.append((a0, b1, a1))
    last_ring = 1 + (latitude_segments - 2) * longitude_segments
    for longitude in range(longitude_segments):
        faces.append(
            (bottom, last_ring + (longitude + 1) % longitude_segments, last_ring + longitude)
        )
    points = np.asarray(vertices, dtype=np.float64)
    points = points @ _rotation_matrix(euler_deg).T
    points += np.asarray(center, dtype=np.float64)
    return points, np.asarray(faces, dtype=np.int64)


def _box_mesh(
    size: tuple[float, float, float], center: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = (value / 2.0 for value in size)
    vertices = np.asarray(
        [
            (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
            (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
        ],
        dtype=np.float64,
    )
    vertices += np.asarray(center, dtype=np.float64)
    faces = np.asarray(
        [
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _torus_mesh(
    center: tuple[float, float, float],
    major_radius: float,
    minor_radius: float,
    *,
    major_segments: int = 32,
    minor_segments: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a torus whose ring lies in the XZ plane."""

    vertices = []
    for major in range(major_segments):
        u = 2.0 * math.pi * major / major_segments
        for minor in range(minor_segments):
            v = 2.0 * math.pi * minor / minor_segments
            radius = major_radius + minor_radius * math.cos(v)
            vertices.append(
                (
                    radius * math.cos(u),
                    minor_radius * math.sin(v),
                    radius * math.sin(u),
                )
            )
    faces = []
    for major in range(major_segments):
        next_major = (major + 1) % major_segments
        for minor in range(minor_segments):
            next_minor = (minor + 1) % minor_segments
            a = major * minor_segments + minor
            b = next_major * minor_segments + minor
            c = next_major * minor_segments + next_minor
            d = major * minor_segments + next_minor
            faces.extend(((a, b, c), (a, c, d)))
    points = np.asarray(vertices, dtype=np.float64)
    points += np.asarray(center, dtype=np.float64)
    return points, np.asarray(faces, dtype=np.int64)


def _part(
    name: str,
    material: str,
    mesh: tuple[np.ndarray, np.ndarray],
) -> MeshPart:
    return MeshPart(name=name, material=material, vertices=mesh[0], faces=mesh[1])


def build_mesh(spec: HandoverObjectSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return the closed plush-body mesh used for collision and PBD smoke tests."""

    return _ellipsoid_mesh(
        spec.semi_axes_m,
        latitude_segments=spec.latitude_segments,
        longitude_segments=spec.longitude_segments,
        irregularity_fraction=spec.irregularity_fraction,
    )


def build_visual_parts(
    spec: HandoverObjectSpec, *, include_accessories: bool
) -> list[MeshPart]:
    """Build the observed hybrid structure; front is +Y and up is +Z."""

    parts = [
        _part("plush_body", "plush_white", build_mesh(spec)),
        _part("black_face_surround", "vinyl_black", _ellipsoid_mesh((0.030, 0.0065, 0.024), center=(0.0, 0.035, 0.006))),
        _part("pink_face_left_lobe", "face_pink", _ellipsoid_mesh((0.0155, 0.0055, 0.0185), center=(-0.0095, 0.041, 0.008))),
        _part("pink_face_right_lobe", "face_pink", _ellipsoid_mesh((0.0155, 0.0055, 0.0185), center=(0.0095, 0.041, 0.008))),
        _part("pink_face_muzzle", "face_pink", _ellipsoid_mesh((0.0240, 0.0058, 0.0120), center=(0.0, 0.042, -0.004))),
        _part("left_eye", "eye_gray", _ellipsoid_mesh((0.0050, 0.0030, 0.0090), center=(-0.0082, 0.0465, 0.013))),
        _part("right_eye", "eye_gray", _ellipsoid_mesh((0.0050, 0.0030, 0.0090), center=(0.0082, 0.0465, 0.013))),
        _part("left_pupil", "vinyl_black", _ellipsoid_mesh((0.0027, 0.0020, 0.0050), center=(-0.0082, 0.0495, 0.0135))),
        _part("right_pupil", "vinyl_black", _ellipsoid_mesh((0.0027, 0.0020, 0.0050), center=(0.0082, 0.0495, 0.0135))),
        _part("nose", "vinyl_black", _ellipsoid_mesh((0.0080, 0.0050, 0.0050), center=(0.0, 0.050, -0.002))),
        _part("viewer_left_ear", "vinyl_black", _ellipsoid_mesh((0.0120, 0.0040, 0.0140), center=(0.031, 0.001, 0.041))),
        _part("viewer_right_ear", "ear_blue", _ellipsoid_mesh((0.0120, 0.0040, 0.0140), center=(-0.031, 0.001, 0.041))),
        _part("left_hand", "plush_white", _ellipsoid_mesh((0.0060, 0.0040, 0.0070), center=(-0.0145, 0.036, -0.020))),
        _part("right_hand", "plush_white", _ellipsoid_mesh((0.0060, 0.0040, 0.0070), center=(0.0145, 0.036, -0.020))),
        _part("left_shoe", "shoe_yellow", _ellipsoid_mesh((0.0100, 0.0070, 0.0045), center=(-0.027, 0.021, -0.037), euler_deg=(0.0, -22.0, 0.0))),
        _part("right_shoe", "shoe_yellow", _ellipsoid_mesh((0.0100, 0.0070, 0.0045), center=(0.027, 0.021, -0.037), euler_deg=(0.0, 22.0, 0.0))),
        _part("graffiti_patch", "graffiti_pink", _ellipsoid_mesh((0.0075, 0.0010, 0.0065), center=(0.034, 0.0050, 0.046))),
        _part("graffiti_drip_1", "graffiti_pink", _ellipsoid_mesh((0.0023, 0.0010, 0.0040), center=(0.031, 0.0052, 0.040))),
        _part("graffiti_drip_2", "graffiti_pink", _ellipsoid_mesh((0.0017, 0.0010, 0.0030), center=(0.026, 0.0052, 0.043))),
        _part("cyan_sprinkle", "sprinkle_cyan", _ellipsoid_mesh((0.0012, 0.0008, 0.0015), center=(0.035, 0.0054, 0.036))),
        _part("yellow_sprinkle", "sprinkle_yellow", _ellipsoid_mesh((0.0012, 0.0008, 0.0015), center=(0.025, 0.0054, 0.049))),
    ]
    for hand_x, prefix in ((-0.0145, "left"), (0.0145, "right")):
        for index, offset in enumerate((-0.0020, 0.0, 0.0020)):
            parts.append(
                _part(
                    f"{prefix}_finger_mark_{index}",
                    "vinyl_black",
                    _ellipsoid_mesh(
                        (0.00065, 0.0006, 0.0012),
                        center=(hand_x + offset, 0.0400, -0.0195),
                        latitude_segments=8,
                        longitude_segments=16,
                    ),
                )
            )
    if include_accessories:
        parts.append(_part("mickey_strap", "vinyl_black", _box_mesh((0.012, 0.0025, 0.050), (0.0, -0.007, 0.061))))
        parts.append(_part("keyring_head", "vinyl_black", _torus_mesh((0.0, -0.007, 0.092), 0.0105, 0.0015)))
        parts.append(_part("keyring_left_ear", "vinyl_black", _torus_mesh((-0.008, -0.007, 0.100), 0.0055, 0.0013, major_segments=24)))
        parts.append(_part("keyring_right_ear", "vinyl_black", _torus_mesh((0.008, -0.007, 0.100), 0.0055, 0.0013, major_segments=24)))
    return parts


def _combine_parts(parts: Iterable[MeshPart]) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    offset = 0
    for part in parts:
        vertices.append(part.vertices)
        faces.append(part.faces + offset)
        offset += len(part.vertices)
    return np.concatenate(vertices), np.concatenate(faces)


def signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return abs(
        float(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            ).sum()
            / 6.0
        )
    )


def _write_mtl(path: Path) -> None:
    lines = ["# Graffiti Mickey procedural material palette"]
    for name, color in MATERIALS.items():
        lines.extend(
            [
                f"newmtl {name}",
                f"Kd {color[0]:.5f} {color[1]:.5f} {color[2]:.5f}",
                "Ka 0.00000 0.00000 0.00000",
                "Ks 0.08000 0.08000 0.08000",
                "Ns 24.00000" if name == "plush_white" else "Ns 180.00000",
                "d 1.00000",
                "illum 2",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_obj(path: Path, parts: Iterable[MeshPart], material_path: Path) -> tuple[np.ndarray, np.ndarray]:
    material_reference = material_path.name
    lines = [
        f"# {ASSET_STEM}",
        "# Front axis +Y; up axis +Z; procedural reference geometry.",
        f"mtllib {material_reference}",
    ]
    offset = 0
    retained = list(parts)
    for part in retained:
        lines.extend((f"o {part.name}", f"usemtl {part.material}"))
        lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in part.vertices)
        lines.extend(
            f"f {a + offset + 1} {b + offset + 1} {c + offset + 1}"
            for a, b, c in part.faces
        )
        offset += len(part.vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _combine_parts(retained)


def write_asset(
    spec: HandoverObjectSpec,
    mesh_path: Path = DEFAULT_MESH,
    manifest_path: Path = DEFAULT_ASSET_MANIFEST,
    *,
    collision_mesh_path: Path | None = None,
    display_mesh_path: Path | None = None,
    material_path: Path | None = None,
) -> dict[str, Any]:
    mesh_path = mesh_path.resolve()
    manifest_path = manifest_path.resolve()
    collision_mesh_path = (
        collision_mesh_path.resolve()
        if collision_mesh_path is not None
        else mesh_path.with_name(f"{mesh_path.stem}_collision.obj")
    )
    display_mesh_path = (
        display_mesh_path.resolve()
        if display_mesh_path is not None
        else mesh_path.with_name(f"{mesh_path.stem}_display.obj")
    )
    material_path = (
        material_path.resolve()
        if material_path is not None
        else mesh_path.with_suffix(".mtl")
    )
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    collision_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    display_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    material_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_mtl(material_path)
    collision_vertices, collision_faces = build_mesh(spec)
    collision_part = MeshPart("plush_body_collision", "plush_white", collision_vertices, collision_faces)
    _write_obj(collision_mesh_path, [collision_part], material_path)
    sim_parts = build_visual_parts(spec, include_accessories=False)
    sim_vertices, sim_faces = _write_obj(mesh_path, sim_parts, material_path)
    display_parts = build_visual_parts(spec, include_accessories=True)
    display_vertices, display_faces = _write_obj(display_mesh_path, display_parts, material_path)
    report = {
        "schema_version": "radeon_oneloop.generated_handover_asset.v2",
        "asset_name": spec.asset_name,
        "exact_sku_status": spec.exact_sku_status,
        "coordinate_convention": {"front": "+Y", "up": "+Z", "viewer_left": "+X"},
        "config_sha256": spec.config_sha256,
        "reference_manifest_sha256": spec.reference_manifest_sha256,
        "materials": {name: list(color) for name, color in MATERIALS.items()},
        "sim_visual": {
            "mesh": mesh_path.name,
            "sha256": sha256_file(mesh_path),
            "vertices": int(len(sim_vertices)),
            "triangles": int(len(sim_faces)),
            "parts": [part.name for part in sim_parts],
            "extents_m": np.ptp(sim_vertices, axis=0).tolist(),
            "accessories_omitted": ["satin strap", "Mickey-head keyring"],
        },
        "display_visual": {
            "mesh": display_mesh_path.name,
            "sha256": sha256_file(display_mesh_path),
            "vertices": int(len(display_vertices)),
            "triangles": int(len(display_faces)),
            "parts": [part.name for part in display_parts],
            "extents_m": np.ptp(display_vertices, axis=0).tolist(),
        },
        "collision": {
            "mesh": collision_mesh_path.name,
            "sha256": sha256_file(collision_mesh_path),
            "vertices": int(len(collision_vertices)),
            "triangles": int(len(collision_faces)),
            "watertight_topology_expected": True,
            "extents_m": np.ptp(collision_vertices, axis=0).tolist(),
            "mesh_volume_m3": signed_mesh_volume(collision_vertices, collision_faces),
            "analytic_ellipsoid_volume_m3": spec.analytic_volume_m3,
        },
        "material_library": {"file": material_path.name, "sha256": sha256_file(material_path)},
        "nominal_overall_height_m": spec.nominal_overall_height_m,
        "overall_height_uncertainty_m": spec.overall_height_uncertainty_m,
        "nominal_mass_kg": spec.nominal_mass_kg,
        "rigid_density_kg_m3": spec.rigid_density_kg_m3,
        "realtime_profile": "rigid convexification of the accessory-free hybrid visual mesh",
        "soft_profile": "plush-body-only Genesis PBD.Elastic proxy; vinyl parts are excluded and material parameters remain qualitative",
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def add_rigid_proxy(
    gs: Any,
    scene: Any,
    spec: HandoverObjectSpec,
    *,
    mesh_path: Path = DEFAULT_MESH,
    pos: tuple[float, float, float],
) -> Any:
    if not mesh_path.is_file():
        raise FileNotFoundError(f"generated handover mesh is missing: {mesh_path}")
    return scene.add_entity(
        gs.morphs.Mesh(
            file=str(mesh_path),
            pos=pos,
            convexify=True,
            decompose_nonconvex=False,
        ),
        material=gs.materials.Rigid(
            rho=spec.rigid_density_kg_m3,
            friction=spec.static_friction,
            coup_restitution=spec.restitution,
        ),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--collision-mesh", type=Path, default=DEFAULT_COLLISION_MESH)
    parser.add_argument("--display-mesh", type=Path, default=DEFAULT_DISPLAY_MESH)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    args = parser.parse_args()
    print(
        json.dumps(
            write_asset(
                load_spec(args.config),
                args.mesh,
                args.manifest,
                collision_mesh_path=args.collision_mesh,
                display_mesh_path=args.display_mesh,
                material_path=args.materials,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
