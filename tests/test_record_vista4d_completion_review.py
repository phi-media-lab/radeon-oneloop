from argparse import Namespace
import json
from pathlib import Path
import tempfile

import pytest

from gaussian.audit_vista4d_completion import AUDIT_SCHEMA, sha256_file
from gaussian.record_vista4d_completion_review import ACCEPTED, CHECKS, build_review


def _fixture(root: Path) -> tuple[Path, Path]:
    proposal = root / "proposal"
    audit = root / "audit"
    proposal.mkdir()
    audit.mkdir()
    (proposal / "manifest.json").write_text("{}\n", encoding="utf-8")
    (proposal / "DONE").write_text("{}\n", encoding="utf-8")
    metrics = {
        "schema_version": AUDIT_SCHEMA,
        "proposal_manifest_sha256": sha256_file(proposal / "manifest.json"),
    }
    (audit / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8"
    )
    (audit / "hashes.sha256").write_text("placeholder\n", encoding="utf-8")
    done = {
        "status": "audit_complete_pending_human_review",
        "metrics_sha256": sha256_file(audit / "metrics.json"),
        "hashes_sha256": sha256_file(audit / "hashes.sha256"),
    }
    (audit / "DONE").write_text(json.dumps(done) + "\n", encoding="utf-8")
    return audit, proposal


def test_acceptance_requires_every_identity_check() -> None:
    with tempfile.TemporaryDirectory() as folder:
        audit, proposal = _fixture(Path(folder))
        values = {name: True for name in CHECKS}
        values[CHECKS[-1]] = False
        args = Namespace(
            audit=audit,
            proposal_run=proposal,
            output=Path(folder) / "review.json",
            decision=ACCEPTED,
            known_defect=["minor texture smear"],
            **values,
        )
        with pytest.raises(ValueError, match="every identity check"):
            build_review(args)


def test_acceptance_is_hash_bound_and_nonformal() -> None:
    with tempfile.TemporaryDirectory() as folder:
        audit, proposal = _fixture(Path(folder))
        args = Namespace(
            audit=audit,
            proposal_run=proposal,
            output=Path(folder) / "review.json",
            decision=ACCEPTED,
            known_defect=["minor texture smear"],
            **{name: True for name in CHECKS},
        )
        result = build_review(args)
        assert result["formal"] is False
        assert result["eligible_for_heldout_real_metrics"] is False
        assert result["allowed_role"] == "low_confidence_generated_training_pseudoviews"
