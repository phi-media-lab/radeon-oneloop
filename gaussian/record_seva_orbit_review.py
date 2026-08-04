#!/usr/bin/env python3
"""Bind an external identity/topology review to one audited SEVA orbit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from gaussian.prepare_four_view_generation import sha256_file
from gaussian.provenance_quarantine import assert_not_quarantined


AUDIT_SCHEMA = "radeon_oneloop.seva_four_view_orbit_audit.v1"
REVIEW_SCHEMA = "radeon_oneloop.seva_four_view_orbit_review.v1"
ACCEPTED = "accepted_low_confidence_pseudoviews"
REJECTED = "rejected"
DECISIONS = (ACCEPTED, REJECTED)
REQUIRED_CHECKS = (
    "single_object_all_views",
    "front_face_confined_to_front_hemisphere",
    "two_asymmetric_ears_correct_sides",
    "rear_strap_and_keyring_stable",
    "no_duplicate_limb_or_floating_surface",
    "four_real_anchor_identities_preserved",
    "adjacent_motion_smooth",
    "cyclic_seam_unobtrusive",
    "background_stable",
    "private_hil_rear_top_identity_consistent",
)
NUMERIC_LIMITS = {
    "anchor_iou_mean_min": 0.55,
    "anchor_iou_min_min": 0.35,
    "adjacent_foreground_iou_p05_min": 0.70,
    "seam_over_adjacent_p95_max": 2.0,
    "foreground_area_cv_max": 0.20,
    "centroid_range_max": 0.25,
}


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


def load_audit(root: Path) -> dict[str, Any]:
    metrics_path = root / "metrics.json"
    done_path = root / "DONE"
    hashes_path = root / "hashes.sha256"
    if not metrics_path.is_file() or not done_path.is_file() or not hashes_path.is_file():
        raise ValueError("SEVA review requires a complete audit")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("unexpected SEVA audit schema")
    if done.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("SEVA audit DONE does not bind metrics")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise ValueError("SEVA audit DONE does not bind hashes")
    for name in ("generated_contact.png", "real_generated_difference_anchors.png"):
        if not (root / name).is_file():
            raise ValueError(f"SEVA audit is missing review evidence: {name}")
    assert_not_quarantined([("seva_audit", metrics)])
    return metrics


def numeric_gates(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    anchor = metrics["real_anchor_silhouette_iou"]
    adjacent = metrics["adjacent_foreground_iou"]
    seam = metrics["cyclic_seam"]
    stability = metrics["foreground_stability"]
    values = {
        "anchor_iou_mean": (float(anchor["mean"]), ">=", NUMERIC_LIMITS["anchor_iou_mean_min"]),
        "anchor_iou_min": (float(anchor["min"]), ">=", NUMERIC_LIMITS["anchor_iou_min_min"]),
        "adjacent_foreground_iou_p05": (
            float(adjacent["p05"]),
            ">=",
            NUMERIC_LIMITS["adjacent_foreground_iou_p05_min"],
        ),
        "seam_over_adjacent_p95": (
            float(seam["rgb_mae_over_adjacent_p95"]),
            "<=",
            NUMERIC_LIMITS["seam_over_adjacent_p95_max"],
        ),
        "foreground_area_cv": (
            float(stability["area_fraction_cv"]),
            "<=",
            NUMERIC_LIMITS["foreground_area_cv_max"],
        ),
        "centroid_x_range": (
            float(stability["centroid_x_range_normalized"]),
            "<=",
            NUMERIC_LIMITS["centroid_range_max"],
        ),
        "centroid_y_range": (
            float(stability["centroid_y_range_normalized"]),
            "<=",
            NUMERIC_LIMITS["centroid_range_max"],
        ),
    }
    return {
        name: {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": value >= threshold if operator == ">=" else value <= threshold,
        }
        for name, (value, operator, threshold) in values.items()
    }


def build_review(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = args.audit_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    metrics = load_audit(audit_root)
    checks = dict(args.check)
    if set(checks) != set(REQUIRED_CHECKS):
        missing = sorted(set(REQUIRED_CHECKS) - set(checks))
        raise ValueError(f"review is missing required checks: {missing}")
    gates = numeric_gates(metrics)
    numeric_pass = all(item["passed"] for item in gates.values())
    if args.decision == ACCEPTED:
        if not all(checks.values()):
            raise ValueError("accepted SEVA pseudo-views must pass every human check")
        if not numeric_pass:
            failed = sorted(name for name, value in gates.items() if not value["passed"])
            raise ValueError(f"accepted SEVA pseudo-views fail numeric gates: {failed}")
        if not args.known_defect:
            raise ValueError("accepted generated pseudo-views must disclose known defects")
    elif all(checks.values()) and numeric_pass:
        raise ValueError("a rejected SEVA orbit must record a failed human or numeric gate")

    review = {
        "schema_version": REVIEW_SCHEMA,
        "created_utc": utc_now(),
        "candidate_id": args.candidate_id,
        "decision": args.decision,
        "accepted_role": "generated_low_confidence_appearance_pseudoviews"
        if args.decision == ACCEPTED
        else None,
        "reviewer_role": "project_agent_visual_review",
        "evidence": {
            "audit_metrics_sha256": sha256_file(audit_root / "metrics.json"),
            "audit_hashes_sha256": sha256_file(audit_root / "hashes.sha256"),
            "generated_contact_sha256": sha256_file(audit_root / "generated_contact.png"),
            "anchor_comparison_sha256": sha256_file(
                audit_root / "real_generated_difference_anchors.png"
            ),
            "seva_manifest_sha256": metrics["seva_manifest_sha256"],
            "four_view_manifest_sha256": metrics["four_view_manifest_sha256"],
            "private_hil_holdout_sha256": sorted(args.private_hil_holdout_sha256),
        },
        "numeric_gates": gates,
        "human_checks": checks,
        "known_defects": args.known_defect,
        "required_downstream_constraints": [
            "initialize_geometry_from_observed_visual_hull_only",
            "freeze_observed_geometry",
            "keep_generated_fill_separate",
            "real_views_have_higher_loss_authority",
            "exclude_generated_views_from_heldout_real_metrics",
        ],
        "prohibited_roles": [
            "observed_view",
            "metric_geometry",
            "collision_geometry",
            "heldout_real_evidence",
            "formal_single_radeon_result",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--decision", choices=DECISIONS, required=True)
    parser.add_argument("--check", action="append", type=parse_check, default=[], required=True)
    parser.add_argument("--known-defect", action="append", default=[])
    parser.add_argument("--private-hil-holdout-sha256", action="append", default=[])
    return parser


def main() -> None:
    print(json.dumps(build_review(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
