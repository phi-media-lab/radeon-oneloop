#!/usr/bin/env python3
"""Record an immutable audit of a generated-fill model and its metric-fit result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "radeon_oneloop.generated_fill_visual_audit.v1"
DONE_SCHEMA_VERSION = "radeon_oneloop.object_asset_stage_done.v1"
ALIGNMENT_MARKER = "one or more SHARP-family-to-VGGT alignment gates failed: "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_complete_run(root: Path) -> dict[str, Any]:
    if not (root / "DONE").is_file() or not (root / "hashes.sha256").is_file():
        raise ValueError(f"generated run is incomplete: {root}")
    for line in (root / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate = (root / relative.lstrip("* ")).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError(f"hash entry escapes generated run: {relative}")
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"generated-run hash mismatch: {candidate}")
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def inspect_failed_alignment(root: Path) -> dict[str, Any]:
    failed_path = root / "FAILED"
    stderr_path = root / "stderr.log"
    if not failed_path.is_file() or not stderr_path.is_file():
        raise ValueError(f"metric-fit diagnostic is not a preserved failed run: {root}")
    stderr = stderr_path.read_text(encoding="utf-8")
    diagnostic = None
    if ALIGNMENT_MARKER in stderr:
        encoded = stderr.rsplit(ALIGNMENT_MARKER, 1)[1].splitlines()[0]
        diagnostic = json.loads(encoded)
    return {
        "run_path": str(root.resolve()),
        "failed_sha256": sha256_file(failed_path),
        "stderr_sha256": sha256_file(stderr_path),
        "alignment_diagnostic": diagnostic,
    }


def write_atomic(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--montage", type=Path, action="append", required=True)
    parser.add_argument("--metric-fit-failure", type=Path, action="append", default=[])
    args = parser.parse_args()

    generator_root = args.generator_run.resolve()
    generator = verify_complete_run(generator_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for reserved in (output / "audit_manifest.json", output / "hashes.sha256", output / "DONE"):
        if reserved.exists():
            raise FileExistsError(f"refusing to overwrite immutable audit evidence: {reserved}")

    montage_records = []
    for source in args.montage:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / source.name
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source, target)
        montage_records.append(
            {"relpath": target.name, "sha256": sha256_file(target), "bytes": target.stat().st_size}
        )

    output_views = [
        {
            "view_id": item.get("view_id"),
            "relpath": item.get("relpath"),
            "sha256": item.get("sha256"),
            "gaussian_count": item.get("gaussian_count"),
        }
        for item in generator.get("outputs", [])
        if str(item.get("relpath", "")).endswith(".ply")
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "formal": False,
        "stage": "generated_fill_candidate",
        "generator": {
            "model": generator.get("model"),
            "run_path": str(generator_root),
            "run_manifest_sha256": sha256_file(generator_root / "manifest.json"),
            "checkpoint_sha256": generator.get("checkpoint_sha256"),
            "hardware": generator.get("hardware"),
            "outputs": output_views,
        },
        "review": {
            "status": "accepted_generated_appearance_proposal_rejected_metric_geometry",
            "appearance_proposal_accepted": True,
            "metric_geometry_accepted": False,
            "identity_consistent_in_local_orbits": True,
            "observed_core_mutated": False,
            "known_limitations": [
                "single-view generated depth is not a metric measurement",
                "local orbit renders do not establish global four-view geometric consistency",
                "generated views may supervise only the separately attributable fill branch",
            ],
        },
        "evaluation_protocol": {
            "kind": "generated_local_orbit_visual_audit_and_metric_alignment_gate",
            "held_out_real": False,
            "eligible_for_heldout_real_metrics": False,
            "eligible_for_formal_metrics": False,
        },
        "montages": montage_records,
        "metric_fit_failures": [
            inspect_failed_alignment(path.resolve()) for path in args.metric_fit_failure
        ],
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
        "schema_version": DONE_SCHEMA_VERSION,
        "status": "done",
        "formal": False,
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_atomic(output / "DONE", json.dumps(done, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
