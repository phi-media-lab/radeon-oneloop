#!/usr/bin/env python3
"""Record an immutable visual audit of observed-core canonicalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not (root / "DONE").is_file() or not (root / "hashes.sha256").is_file():
        raise ValueError(f"run is incomplete: {root}")
    for line in (root / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate = (root / relative.lstrip("* ")).resolve()
        if not candidate.is_relative_to(root) or sha256_file(candidate) != expected:
            raise ValueError(f"run hash mismatch: {candidate}")
    manifest_path = root / "manifest.json"
    return {
        "path": str(root),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "manifest_sha256": sha256_file(manifest_path),
    }


def write_atomic(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--canonical-render-run", type=Path, required=True)
    parser.add_argument("--canonical-provenance", type=Path, required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = inspect_run(args.source_run)
    render = inspect_run(args.canonical_render_run)
    provenance_path = args.canonical_provenance.resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_ply = args.source_run.resolve() / "train" / "splat.ply"
    if provenance.get("provenance_class") != "observed_core_candidate":
        raise ValueError("canonical provenance is not observed-core evidence")
    if provenance.get("input_ply_sha256") != sha256_file(source_ply):
        raise ValueError("canonical provenance does not match source observed PLY")
    if provenance.get("output_ply_sha256") != render["manifest"].get("ply_sha256"):
        raise ValueError("canonical render does not match canonical observed PLY")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for reserved in (output / "audit_manifest.json", output / "hashes.sha256", output / "DONE"):
        if reserved.exists():
            raise FileExistsError(f"refusing to overwrite immutable audit evidence: {reserved}")
    montage_source = args.montage.resolve()
    montage_target = output / montage_source.name
    shutil.copy2(montage_source, montage_target)
    manifest = {
        "schema_version": "radeon_oneloop.observed_core_canonical_visual_audit.v1",
        "formal": False,
        "stage": "observed_appearance_core",
        "review": {
            "status": "accepted_canonical_coordinate_conversion",
            "identity_consistent_across_four_observed_directions": True,
            "metric_camera_render_consistent": True,
            "observed_appearance_mutated": False,
            "known_limitations": [
                "all four anchor images were training views",
                "this audit validates coordinate conversion and rendering, not held-out quality",
                "hard-mask halo remains visible around fur boundaries",
            ],
        },
        "evaluation_protocol": {
            "kind": "observed_training_view_canonical_reprojection_visual_audit",
            "held_out": False,
            "eligible_for_formal_metrics": False,
        },
        "source_run": {
            "path": source["path"],
            "manifest_sha256": source["manifest_sha256"],
            "splat_sha256": sha256_file(source_ply),
        },
        "canonicalization": {
            "provenance_sha256": sha256_file(provenance_path),
            "output_ply_sha256": provenance["output_ply_sha256"],
            "inverse_similarity": provenance["inverse_similarity_normalized_to_canonical"],
        },
        "canonical_render": {
            "path": render["path"],
            "manifest_sha256": render["manifest_sha256"],
        },
        "montage": {
            "relpath": montage_target.name,
            "sha256": sha256_file(montage_target),
            "layout": "front,right,rear,left canonical metric-camera renders",
        },
    }
    manifest_path = output / "audit_manifest.json"
    write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hash_lines = [
        f"{sha256_file(path)}  {path.name}\n"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}
    ]
    write_atomic(output / "hashes.sha256", "".join(hash_lines))
    done = {
        "schema_version": "radeon_oneloop.object_asset_stage_done.v1",
        "formal": False,
        "status": "done",
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_atomic(output / "DONE", json.dumps(done, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
