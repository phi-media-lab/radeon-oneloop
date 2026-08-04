#!/usr/bin/env python3
"""Concatenate an observed Gaussian core and an independently pruned fill layer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from gaussian.gaussian_appearance_delta import parse_vertex_layout, sha256_file
from gaussian.provenance_quarantine import assert_not_quarantined


SCHEMA = "radeon_oneloop.layered_gaussian_appearance_fusion.v1"
DONE_SCHEMA = "radeon_oneloop.layered_gaussian_appearance_fusion_done.v1"


class LayerFusionError(ValueError):
    """Raised when observed and generated appearance layers cannot be fused safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_provenance(path: Path, expected_ply: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("output_ply_sha256") != sha256_file(expected_ply):
        raise LayerFusionError(f"provenance does not bind its PLY: {path}")
    return value


def _header(path: Path) -> list[str]:
    lines = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise LayerFusionError(f"PLY header is incomplete: {path}")
            lines.append(raw.decode("ascii"))
            if raw.strip() == b"end_header":
                return lines


def _vertices(path: Path) -> tuple[int, np.dtype, np.memmap]:
    offset, count, dtype = parse_vertex_layout(path)
    return count, dtype, np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def _write_fused(
    observed_path: Path,
    output: Path,
    observed: np.memmap,
    fill: np.memmap,
) -> None:
    header = _header(observed_path)
    total = len(observed) + len(fill)
    rewritten = []
    replaced = False
    for line in header:
        if line.strip() == f"element vertex {len(observed)}":
            rewritten.append(f"element vertex {total}\n")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        raise LayerFusionError("observed PLY vertex count line is missing")
    with output.open("xb") as handle:
        handle.write("".join(rewritten).encode("ascii"))
        handle.write(np.asarray(observed).tobytes())
        handle.write(np.asarray(fill).tobytes())


def fuse(args: argparse.Namespace) -> dict[str, Any]:
    observed_ply = args.observed_ply.resolve()
    observed_provenance_path = args.observed_provenance.resolve()
    fill_ply = args.fill_ply.resolve()
    fill_provenance_path = args.fill_provenance.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    observed_provenance = _load_provenance(observed_provenance_path, observed_ply)
    fill_provenance = _load_provenance(fill_provenance_path, fill_ply)
    if observed_provenance.get("provenance_class") != "observed_core_candidate":
        raise LayerFusionError("observed source is not an observed-core candidate")
    if fill_provenance.get("provenance_class") != "generated_fill_candidate":
        raise LayerFusionError("fill source is not a generated-fill candidate")
    if observed_provenance.get("observed_only_training") is not True:
        raise LayerFusionError("observed core was not trained from observed-only evidence")
    if fill_provenance.get("formal") is not False:
        raise LayerFusionError("generated fill must remain nonformal")
    if not fill_provenance.get("observed_visibility_prune_metrics_sha256"):
        raise LayerFusionError("generated fill has not passed observed-visibility pruning")
    assert_not_quarantined(
        [
            ("observed_core_provenance", observed_provenance),
            ("generated_fill_provenance", fill_provenance),
        ]
    )

    observed_count, observed_dtype, observed_vertices = _vertices(observed_ply)
    fill_count, fill_dtype, fill_vertices = _vertices(fill_ply)
    if observed_dtype != fill_dtype:
        raise LayerFusionError("observed and fill PLY layouts differ")
    if observed_count < 1000 or fill_count < 1:
        raise LayerFusionError("appearance layers contain too few Gaussians")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        fused_ply = staging / "appearance_fused_preview.ply"
        _write_fused(observed_ply, fused_ply, observed_vertices, fill_vertices)
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "provenance_class": "confidence_fused_candidate",
            "observed_core": {
                "ply_sha256": sha256_file(observed_ply),
                "provenance_sha256": sha256_file(observed_provenance_path),
                "gaussian_count": observed_count,
                "source_formal": observed_provenance.get("formal"),
                "authoritative_on_observed_support": True,
            },
            "generated_fill": {
                "ply_sha256": sha256_file(fill_ply),
                "provenance_sha256": sha256_file(fill_provenance_path),
                "gaussian_count": fill_count,
                "visibility_prune_metrics_sha256": fill_provenance[
                    "observed_visibility_prune_metrics_sha256"
                ],
                "authoritative_on_observed_support": False,
            },
            "fusion": {
                "method": "binary_PLY_vertex_concatenation_no_geometry_blending",
                "observed_core_preserved_bitwise": True,
                "generated_fill_toggleable": True,
                "total_gaussians": observed_count + fill_count,
                "output_ply_sha256": sha256_file(fused_ply),
            },
            "allowed_role": "nonformal_layered_appearance_preview",
            "prohibited_roles": [
                "observed_only_asset",
                "formal_single_radeon_result",
                "collision_geometry",
                "heldout_real_evidence",
            ],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance = {
            "schema_version": "radeon_oneloop.layered_gaussian_provenance.v1",
            "formal": False,
            "host_role": args.host_role,
            "provenance_class": "confidence_fused_candidate",
            "observed_only_training": False,
            "output_ply_sha256": sha256_file(fused_ply),
            "gaussian_count": observed_count + fill_count,
            "layer_manifest_sha256": sha256_file(manifest_path),
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
        }
        provenance_path = staging / "appearance_fused_preview.provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = []
        for path in sorted(staging.iterdir()):
            if path.is_file() and path.name not in {"hashes.sha256", "DONE"}:
                lines.append(f"{sha256_file(path)}  {path.name}")
        hashes_path = staging / "hashes.sha256"
        hashes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": DONE_SCHEMA,
                    "status": "done_nonformal_layered_appearance_candidate",
                    "manifest_sha256": sha256_file(manifest_path),
                    "provenance_sha256": sha256_file(provenance_path),
                    "hashes_sha256": sha256_file(hashes_path),
                    "completed_utc": utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return manifest
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps(
                {
                    "schema_version": "radeon_oneloop.layered_gaussian_appearance_fusion_failure.v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_utc": utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failed = output.with_name(f"{output.name}.FAILED")
        if not failed.exists():
            os.replace(staging, failed)
        else:
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-ply", type=Path, required=True)
    parser.add_argument("--observed-provenance", type=Path, required=True)
    parser.add_argument("--fill-ply", type=Path, required=True)
    parser.add_argument("--fill-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-role", default="radeon_f_gpu0_gfx1100_nonformal")
    return parser


def main() -> None:
    print(json.dumps(fuse(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
