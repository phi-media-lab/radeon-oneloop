#!/usr/bin/env python3
"""Record a hash-bound human review of one audited Vista4D proposal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from gaussian.audit_vista4d_completion import AUDIT_SCHEMA, sha256_file


REVIEW_SCHEMA = "radeon_oneloop.vista4d_completion_human_review.v1"
ACCEPTED = "accepted_for_low_confidence_pseudoviews"
REJECTED = "rejected_for_pseudoview_training"
CHECKS = (
    "single_object",
    "stable_two_ears",
    "stable_face_identity",
    "stable_backstrap",
    "no_duplicate_parts",
    "loop_coherent",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_audit(audit_root: Path, proposal_root: Path) -> dict[str, Any]:
    metrics_path = audit_root / "metrics.json"
    hashes_path = audit_root / "hashes.sha256"
    done_path = audit_root / "DONE"
    proposal_path = proposal_root / "manifest.json"
    for path in (metrics_path, hashes_path, done_path, proposal_path, proposal_root / "DONE"):
        if not path.is_file():
            raise ValueError(f"review input is incomplete: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("only the mask-bound Vista4D audit v2 may be reviewed")
    if done.get("status") != "audit_complete_pending_human_review":
        raise ValueError("Vista4D audit is not reviewable")
    if done.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("Vista4D audit DONE does not bind metrics")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise ValueError("Vista4D audit DONE does not bind its hash index")
    proposal_sha = sha256_file(proposal_path)
    if metrics.get("proposal_manifest_sha256") != proposal_sha:
        raise ValueError("audit and proposal lineage differ")
    return metrics


def build_review(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = args.audit.resolve()
    proposal_root = args.proposal_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    metrics = validate_audit(audit_root, proposal_root)
    checks = {name: bool(getattr(args, name)) for name in CHECKS}
    if args.decision == ACCEPTED and not all(checks.values()):
        missing = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"accepted pseudo-views require every identity check: {missing}")
    if args.decision == ACCEPTED and not args.known_defect:
        raise ValueError("accepted generated pseudo-views require explicit known defects")
    value = {
        "schema_version": REVIEW_SCHEMA,
        "created_utc": utc_now(),
        "formal": False,
        "eligible_for_heldout_real_metrics": False,
        "decision": args.decision,
        "proposal_manifest_sha256": metrics["proposal_manifest_sha256"],
        "audit_metrics_sha256": sha256_file(audit_root / "metrics.json"),
        "audit_hash_index_sha256": sha256_file(audit_root / "hashes.sha256"),
        "checks": checks,
        "known_defects": list(args.known_defect),
        "allowed_role": (
            "low_confidence_generated_training_pseudoviews"
            if args.decision == ACCEPTED
            else None
        ),
        "prohibited_roles": [
            "observed_evidence",
            "heldout_real_evidence",
            "metric_geometry_truth",
            "physics_collision_geometry",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--proposal-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", choices=(ACCEPTED, REJECTED), required=True)
    for name in CHECKS:
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_true")
    parser.add_argument("--known-defect", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_review(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
