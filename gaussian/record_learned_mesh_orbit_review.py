#!/usr/bin/env python3
"""Bind a human topology review to one generated learned-mesh orbit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from gaussian.prepare_four_view_generation import sha256_file
from gaussian.provenance_quarantine import assert_not_quarantined


TEXTURE_SCHEMA = "radeon_oneloop.four_view_learned_mesh_texture_orbit.v2"
REVIEW_SCHEMA = "radeon_oneloop.learned_mesh_orbit_visual_review.v1"
ACCEPTED = "accepted_conditioning_only"
SUPERSEDED = "superseded_valid_candidate"
REJECTED = "rejected"
DECISIONS = (ACCEPTED, SUPERSEDED, REJECTED)
REQUIRED_CHECKS = (
    "single_front_face",
    "face_absent_from_rear_hemisphere",
    "two_asymmetric_ears_correct_sides",
    "single_rear_strap",
    "continuous_oval_body_no_duplicate_surface",
    "four_cardinal_silhouettes_pass",
    "private_hil_rear_top_identity_consistent",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_check(value: str) -> tuple[str, bool]:
    try:
        name, raw = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checks must use NAME=true|false") from exc
    if name not in REQUIRED_CHECKS or raw not in {"true", "false"}:
        raise argparse.ArgumentTypeError(f"unsupported review check: {value}")
    return name, raw == "true"


def load_texture(texture_root: Path) -> dict[str, Any]:
    manifest_path = texture_root / "manifest.json"
    done_path = texture_root / "DONE"
    if not manifest_path.is_file() or not done_path.is_file():
        raise ValueError("texture orbit requires manifest.json and DONE")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TEXTURE_SCHEMA:
        raise ValueError("unexpected learned-mesh texture schema")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("texture DONE does not bind its manifest")
    if done.get("hashes_sha256") != sha256_file(texture_root / "hashes.sha256"):
        raise ValueError("texture DONE does not bind its hash index")
    contact_sheet = texture_root / "audit/orbit_contact_sheet.png"
    if not contact_sheet.is_file():
        raise ValueError("texture orbit is missing its visual contact sheet")
    for relpath, expected in (
        (manifest["orbit"]["source_video_relpath"], manifest["orbit"]["source_video_sha256"]),
        (manifest["mesh"]["ply_relpath"], manifest["mesh"]["ply_sha256"]),
    ):
        path = texture_root / relpath
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"texture review evidence hash mismatch: {relpath}")
    assert_not_quarantined([("learned_mesh_texture_manifest", manifest)])
    return manifest


def build_review(args: argparse.Namespace) -> dict[str, Any]:
    texture_root = args.texture_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = load_texture(texture_root)
    checks = dict(args.check)
    if set(checks) != set(REQUIRED_CHECKS):
        missing = sorted(set(REQUIRED_CHECKS) - set(checks))
        raise ValueError(f"review is missing required checks: {missing}")
    if args.decision in {ACCEPTED, SUPERSEDED} and not all(checks.values()):
        raise ValueError("an accepted or superseded-valid candidate must pass every check")
    if args.decision == REJECTED and all(checks.values()):
        raise ValueError("a rejected candidate must record at least one failed check")
    if args.decision == ACCEPTED and not args.known_defect:
        raise ValueError("accepted generated conditioning must disclose known defects")

    contact_sheet = texture_root / "audit/orbit_contact_sheet.png"
    value = {
        "schema_version": REVIEW_SCHEMA,
        "created_utc": utc_now(),
        "candidate_id": args.candidate_id,
        "decision": args.decision,
        "accepted_role": "generated_conditioning_only" if args.decision == ACCEPTED else None,
        "reviewer_role": "project_agent_visual_review",
        "evidence": {
            "texture_manifest_sha256": sha256_file(texture_root / "manifest.json"),
            "orbit_contact_sheet_sha256": sha256_file(contact_sheet),
            "source_video_sha256": manifest["orbit"]["source_video_sha256"],
            "aligned_mesh_sha256": manifest["input"]["aligned_learned_mesh_sha256"],
            "four_view_manifest_sha256": manifest["input"]["four_view_manifest_sha256"],
            "private_hil_holdout_sha256": sorted(args.private_hil_holdout_sha256),
        },
        "checks": checks,
        "known_defects": args.known_defect,
        "prohibited_roles": [
            "observed_geometry",
            "final_metric_geometry",
            "collision_geometry",
            "heldout_real_evidence",
            "formal_single_radeon_lineage",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--decision", choices=DECISIONS, required=True)
    parser.add_argument("--check", action="append", type=parse_check, default=[], required=True)
    parser.add_argument("--known-defect", action="append", default=[])
    parser.add_argument("--private-hil-holdout-sha256", action="append", default=[])
    return parser


def main() -> None:
    value = build_review(build_parser().parse_args())
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
