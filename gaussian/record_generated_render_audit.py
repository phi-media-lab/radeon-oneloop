#!/usr/bin/env python3
"""Record an immutable visual decision for a nonformal generated-fill render."""

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
        raise ValueError(f"render run is incomplete: {root}")
    for line in (root / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate = (root / relative.lstrip("* ")).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"hash entry escapes render run: {relative}")
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"render-run hash mismatch: {candidate}")
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
    parser.add_argument("--render-run", type=Path, required=True)
    parser.add_argument("--baseline-render-run", type=Path)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--montage-layout",
        default="top=baseline render; bottom=candidate render; columns=front,right,rear,left",
    )
    parser.add_argument(
        "--decision",
        choices=("rejected_direct_appearance_fill", "accepted_for_confidence_pruning_not_final"),
        default="rejected_direct_appearance_fill",
    )
    args = parser.parse_args()

    candidate = inspect_run(args.render_run)
    baseline = inspect_run(args.baseline_render_run) if args.baseline_render_run else None
    provenance_path = args.source_provenance.resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("eligible_for_formal_metrics") is not False:
        raise ValueError("generated source provenance must explicitly reject formal metrics")
    if candidate["manifest"].get("ply_sha256") != provenance.get("output_ply_sha256"):
        raise ValueError("rendered PLY does not match source provenance")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for reserved in (output / "audit_manifest.json", output / "hashes.sha256", output / "DONE"):
        if reserved.exists():
            raise FileExistsError(f"refusing to overwrite immutable audit evidence: {reserved}")
    montage_source = args.montage.resolve()
    montage_target = output / montage_source.name
    shutil.copy2(montage_source, montage_target)

    pruning_candidate = args.decision == "accepted_for_confidence_pruning_not_final"
    reasons = (
        [
            "single-source support substantially improves exterior continuity in all canonical directions",
            "the object remains recognizable from front, sides, and rear",
            "duplicated ear layers and peripheral floaters still prevent direct release",
            "single-source Gaussians must be limited to regions not covered by observed or cross-source geometry",
        ]
        if pruning_candidate
        else [
            "large transmittance holes or structured sparsity remain in all canonical directions",
            "appearance-field donation or global voxel reduction does not repair missing exterior geometry",
            "generated components require a separately attributable confidence-masked fill branch",
        ]
    )
    manifest = {
        "schema_version": "radeon_oneloop.generated_fill_render_visual_audit.v1",
        "formal": False,
        "stage": "generated_fill_candidate",
        "review": {
            "status": args.decision,
            "direct_appearance_fill_accepted": False,
            "accepted_for_confidence_pruning": pruning_candidate,
            "geometry_prior_evidence_retained": True,
            "appearance_pseudoview_branch_retained": True,
            "observed_core_mutated": False,
            "reasons": reasons,
        },
        "evaluation_protocol": {
            "kind": "nonformal_RADV_VkSplat_visual_comparison",
            "held_out_real": False,
            "eligible_for_heldout_real_metrics": False,
            "eligible_for_formal_metrics": False,
        },
        "candidate_render": {
            "run_path": candidate["path"],
            "run_manifest_sha256": candidate["manifest_sha256"],
            "ply_sha256": candidate["manifest"]["ply_sha256"],
            "gaussian_count": candidate["manifest"]["gaussian_count"],
        },
        "baseline_render": (
            {
                "run_path": baseline["path"],
                "run_manifest_sha256": baseline["manifest_sha256"],
                "ply_sha256": baseline["manifest"]["ply_sha256"],
            }
            if baseline is not None
            else None
        ),
        "source_provenance_sha256": sha256_file(provenance_path),
        "montage": {
            "relpath": montage_target.name,
            "sha256": sha256_file(montage_target),
            "layout": args.montage_layout,
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
