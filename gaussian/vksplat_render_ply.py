#!/usr/bin/env python3
"""Render a standard 3DGS PLY at object cameras through VkSplat/RADV."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}
REQUIRED_PROPERTIES = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)


class VkSplatRenderError(RuntimeError):
    pass


def validate_source_provenance(
    source_provenance: dict[str, object], *, formal: bool, host_role: str
) -> None:
    if source_provenance.get("formal") is not formal:
        raise VkSplatRenderError("source provenance formal flag does not match the render")
    if formal:
        if host_role != "radeon_c_gpu0_gfx1100_formal":
            raise VkSplatRenderError("formal rendering requires the Radeon-c formal host role")
        lineage = source_provenance.get("training_lineage")
        if not isinstance(lineage, dict) or lineage.get("training_formal") is not True:
            raise VkSplatRenderError("formal rendering requires formal training lineage")
        if lineage.get("secondary_accelerator_artifacts") is not False:
            raise VkSplatRenderError("formal rendering rejects secondary-accelerator artifacts")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_binary_vertex_header(handle: BinaryIO) -> tuple[int, np.dtype]:
    if handle.readline() != b"ply\n":
        raise VkSplatRenderError("not a PLY file")
    if handle.readline().decode("ascii").strip() != "format binary_little_endian 1.0":
        raise VkSplatRenderError("only binary_little_endian PLY is supported")
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    active_element: str | None = None
    while True:
        raw = handle.readline()
        if not raw:
            raise VkSplatRenderError("PLY header ended before end_header")
        line = raw.decode("ascii").strip()
        if line == "end_header":
            break
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "element":
            if len(parts) != 3:
                raise VkSplatRenderError(f"invalid PLY element line: {line}")
            active_element = parts[1]
            if active_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and active_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise VkSplatRenderError("list-valued vertex properties are unsupported")
            if parts[1] not in PLY_TYPES:
                raise VkSplatRenderError(f"unsupported PLY scalar type: {parts[1]}")
            vertex_properties.append((parts[2], PLY_TYPES[parts[1]]))
    if vertex_count is None or vertex_count <= 0:
        raise VkSplatRenderError("positive vertex count is required")
    dtype = np.dtype(vertex_properties, align=False)
    if not set(REQUIRED_PROPERTIES).issubset(dtype.names or ()):
        missing = sorted(set(REQUIRED_PROPERTIES) - set(dtype.names or ()))
        raise VkSplatRenderError(f"PLY is missing 3DGS properties: {missing}")
    return vertex_count, dtype


def read_3dgs_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        count, dtype = parse_binary_vertex_header(handle)
        payload = handle.read(count * dtype.itemsize)
    if len(payload) != count * dtype.itemsize:
        raise VkSplatRenderError("PLY vertex payload is truncated")
    vertices = np.frombuffer(payload, dtype=dtype, count=count)
    xyz = np.ascontiguousarray(np.stack([vertices[name] for name in ("x", "y", "z")], axis=1), dtype=np.float32)
    rotations = np.ascontiguousarray(
        np.stack([vertices[f"rot_{index}"] for index in range(4)], axis=1), dtype=np.float32
    )
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1.0e-12)
    scale_logits = np.stack([vertices[f"scale_{index}"] for index in range(3)], axis=1)
    scales = np.ascontiguousarray(np.exp(scale_logits), dtype=np.float32)
    logits = np.asarray(vertices["opacity"], dtype=np.float32)
    opacities = np.ascontiguousarray((1.0 / (1.0 + np.exp(-logits)))[:, None], dtype=np.float32)
    sh = np.zeros((count, 16, 3), dtype=np.float32)
    sh[:, 0, :] = np.stack([vertices[f"f_dc_{index}"] for index in range(3)], axis=1)
    for coefficient in range(1, 16):
        for channel in range(3):
            name = f"f_rest_{channel * 15 + coefficient - 1}"
            if name in vertices.dtype.names:
                sh[:, coefficient, channel] = vertices[name]
    return {
        "xyz": xyz,
        "rotations": rotations,
        "scales": scales,
        "opacities": opacities,
        "sh": np.ascontiguousarray(sh),
    }


def write_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise VkSplatRenderError(f"failed to write {path}")


def render(args: argparse.Namespace) -> dict[str, object]:
    import sys

    vksplat_root = args.vksplat_root.resolve()
    sys.path.insert(0, str(vksplat_root / "vksplat"))
    import vksplat  # type: ignore  # noqa: WPS433

    ply_path = args.ply.resolve()
    cameras_path = args.cameras.resolve()
    source_provenance_path = args.source_provenance.resolve()
    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    validate_source_provenance(
        source_provenance, formal=bool(args.formal), host_role=args.host_role
    )
    if source_provenance.get("provenance_class") not in {
        "generated_fill_candidate",
        "observed_core_candidate",
        "confidence_fused_candidate",
    }:
        raise VkSplatRenderError("renderer received an unsupported provenance class")
    if source_provenance.get("output_ply_sha256") != sha256_file(ply_path):
        raise VkSplatRenderError("PLY does not match its source provenance")

    cameras_document = json.loads(cameras_path.read_text(encoding="utf-8"))
    if cameras_document.get("camera_model") != "PINHOLE_OPENCV":
        raise VkSplatRenderError("only PINHOLE_OPENCV cameras are supported")
    cameras = cameras_document["cameras"]
    if len(cameras) != 4:
        raise VkSplatRenderError("object audit requires exactly four cameras")
    gaussians = read_3dgs_ply(ply_path)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    module = vksplat.VkSplat()
    shader_dir = str((vksplat_root / "vksplat" / "shader").resolve()) + "//"
    module.initialize(shader_dir, -1)
    try:
        # The pinned binding's py::arg names are stale. Positional order follows
        # the compiled C++ signature: xyz, rotations, scales, opacities, SH.
        module.set_gauss_params(
            gaussians["xyz"],
            gaussians["rotations"],
            gaussians["scales"],
            gaussians["opacities"],
            gaussians["sh"],
        )
        records = []
        for camera in cameras:
            width, height = (int(value) for value in camera["image_size_wh"])
            intrinsic = np.asarray(camera["intrinsic_3x3"], dtype=np.float32)
            world_to_camera = np.asarray(camera["world_to_camera_opencv_4x4"], dtype=np.float32)
            module.set_uniforms(
                args.active_sh_degree,
                world_to_camera,
                height,
                width,
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
                False,
            )
            module.forward()
            pixel_state = np.asarray(module.pixel_state).copy()
            if pixel_state.shape != (height, width, 4):
                raise VkSplatRenderError(f"unexpected VkSplat output shape: {pixel_state.shape}")
            transmittance = np.clip(pixel_state[..., 3:4], 0.0, 1.0)
            rgb = np.clip(pixel_state[..., :3] + transmittance * args.background, 0.0, 1.0)
            rgb_u8 = np.round(rgb * 255.0).astype(np.uint8)
            image_path = output / f"{camera['view_id']}.png"
            write_png(image_path, rgb_u8)
            records.append(
                {
                    "view_id": camera["view_id"],
                    "path": image_path.name,
                    "sha256": sha256_file(image_path),
                    "width": width,
                    "height": height,
                    "mean_transmittance": float(transmittance.mean()),
                }
            )
        device = module.get_vram_usage()
        peak = module.get_peak_vram_usage()
    finally:
        module.cleanup()

    purpose_by_provenance = {
        "observed_core_candidate": "nonformal observed-core visual QA",
        "generated_fill_candidate": "nonformal generated-fill visual QA",
        "confidence_fused_candidate": "nonformal confidence-fused visual QA",
    }
    purpose = purpose_by_provenance[source_provenance["provenance_class"]]
    if args.formal:
        purpose = purpose.replace("nonformal", "formal")
    manifest = {
        "schema_version": "radeon_oneloop.vksplat_generated_ply_render.v1",
        "formal": bool(args.formal),
        "host_role": args.host_role,
        "renderer": "VkSplat_RADV",
        "vksplat_commit": args.vksplat_commit,
        "ply_sha256": sha256_file(ply_path),
        "cameras_sha256": sha256_file(cameras_path),
        "source_provenance_sha256": sha256_file(source_provenance_path),
        "gaussian_count": int(len(gaussians["xyz"])),
        "active_sh_degree": args.active_sh_degree,
        "background": args.background,
        "vram_bytes": int(device),
        "peak_vram_bytes": int(peak),
        "renders": records,
        "purpose": purpose,
        "eligible_for_formal_metrics": bool(args.formal),
        "eligible_for_heldout_real_metrics": False,
    }
    (output / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--vksplat-commit", required=True)
    parser.add_argument("--active-sh-degree", type=int, default=0, choices=range(4))
    parser.add_argument("--background", type=float, default=0.125)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--host-role", default="unspecified_nonformal")
    return parser.parse_args()


def main() -> int:
    render(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
