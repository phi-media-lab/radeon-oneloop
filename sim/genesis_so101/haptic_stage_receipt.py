#!/usr/bin/env python3
"""Seal machine and operator evidence for staged haptic progression.

The physical run directory remains immutable.  This command creates a separate
content-addressed receipt that binds its accepted machine gate and hash index
to a constrained post-run operator verdict.  Only an accepted single-joint
receipt can authorize the next monitor-only stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from datetime import datetime, timezone


SCHEMA_VERSION = "radeon_oneloop.haptic_stage_receipt.v1"
MONITOR_RECEIPT_SCHEMA_VERSION = "radeon_oneloop.haptic_monitor_receipt.v1"
ARM_PHYSICAL_RECEIPT_SCHEMA_VERSION = (
    "radeon_oneloop.haptic_arm_physical_receipt.v1"
)
PERCEPTION_CHOICES = (
    "useful_comfortable",
    "too_weak",
    "too_strong",
    "unsafe_or_uncomfortable",
)
MONITOR_MAPPING_CHOICES = (
    "correct_same_side_same_direction",
    "wrong_side_or_direction",
    "unstable_or_unusable",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_index_by_basename(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise ValueError("source hash index contains an invalid line")
        basename = Path(fields[1].lstrip("*")).name
        if basename in result:
            raise ValueError(f"source hash index repeats basename {basename!r}")
        result[basename] = fields[0]
    return result


def build_single_joint_receipt(
    *,
    source_run_id: str,
    gate_path: Path,
    source_hash_index_path: Path,
    source_done_path: Path,
    perception: str,
    leader_moves_freely_after_test: bool,
) -> dict[str, Any]:
    if RUN_ID_PATTERN.fullmatch(source_run_id) is None:
        raise ValueError("source_run_id contains unsafe characters")
    if perception not in PERCEPTION_CHOICES:
        raise ValueError(f"unsupported perception verdict: {perception!r}")
    for path in (gate_path, source_hash_index_path, source_done_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_sha256 = sha256_file(gate_path)
    hashes = _hash_index_by_basename(source_hash_index_path)
    checks = {
        "machine_gate_schema": gate.get("schema_version")
        == "radeon_oneloop.haptic_bench_gate.v1",
        "machine_gate_accepted": gate.get("accepted") is True,
        "machine_gate_physical_output_recorded": (
            gate.get("physical_output_commands") is True
        ),
        "machine_gate_waited_for_operator": (
            gate.get("operator_perception_gate") == "pending_separate_attestation"
        ),
        "gate_bound_by_source_hash_index": hashes.get(gate_path.name)
        == gate_sha256,
        "source_done_marker_present": source_done_path.is_file(),
        "operator_perception_useful_comfortable": (
            perception == "useful_comfortable"
        ),
        "operator_reports_free_motion_after_shutdown": (
            leader_moves_freely_after_test is True
        ),
    }
    accepted = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "formal": False,
        "stage": "single_joint_physical",
        "accepted": accepted,
        "checks": checks,
        "source": {
            "run_id": source_run_id,
            "gate_sha256": gate_sha256,
            "hash_index_sha256": sha256_file(source_hash_index_path),
            "done_marker_sha256": sha256_file(source_done_path),
        },
        "operator_attestation": {
            "perception": perception,
            "leader_moves_freely_after_test": leader_moves_freely_after_test,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "operator_identity_recorded": False,
        },
        "next_authorized_stage": (
            "single_arm_monitor_only" if accepted else None
        ),
        "physical_output_commands": False,
        "receipt_writes_to_source_run": False,
    }


def build_single_arm_monitor_receipt(
    *,
    source_run_id: str,
    gate_path: Path,
    source_hash_index_path: Path,
    source_done_path: Path,
    mapping_verdict: str,
    leader_moves_freely_after_monitor: bool,
) -> dict[str, Any]:
    """Bind the operator's mapping judgment to an accepted monitor run."""

    if RUN_ID_PATTERN.fullmatch(source_run_id) is None:
        raise ValueError("source_run_id contains unsafe characters")
    if mapping_verdict not in MONITOR_MAPPING_CHOICES:
        raise ValueError(f"unsupported mapping verdict: {mapping_verdict!r}")
    for path in (gate_path, source_hash_index_path, source_done_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_sha256 = sha256_file(gate_path)
    hashes = _hash_index_by_basename(source_hash_index_path)
    done = json.loads(source_done_path.read_text(encoding="utf-8"))
    checks = {
        "machine_gate_schema": (
            gate.get("schema_version")
            == "radeon_oneloop.haptic_monitor_gate.v1"
        ),
        "machine_gate_accepted": gate.get("accepted") is True,
        "machine_gate_stage": gate.get("stage") == "single_arm_monitor_only",
        "machine_gate_nonphysical": (
            gate.get("physical_output_commands") is False
        ),
        "machine_gate_waited_for_operator": (
            gate.get("next_stage_requires_operator_receipt") is True
        ),
        "gate_bound_by_source_hash_index": hashes.get(gate_path.name)
        == gate_sha256,
        "source_done_marker_accepted": (
            done.get("status") == "done_single_arm_monitor_machine_accepted"
        ),
        "operator_confirms_same_side_same_direction": (
            mapping_verdict == "correct_same_side_same_direction"
        ),
        "operator_reports_free_motion_after_monitor": (
            leader_moves_freely_after_monitor is True
        ),
    }
    accepted = all(checks.values())
    return {
        "schema_version": MONITOR_RECEIPT_SCHEMA_VERSION,
        "formal": False,
        "stage": "single_arm_monitor_only",
        "selected_side": gate.get("selected_side"),
        "accepted": accepted,
        "checks": checks,
        "source": {
            "run_id": source_run_id,
            "gate_sha256": gate_sha256,
            "hash_index_sha256": sha256_file(source_hash_index_path),
            "done_marker_sha256": sha256_file(source_done_path),
        },
        "operator_attestation": {
            "mapping_verdict": mapping_verdict,
            "leader_moves_freely_after_monitor": (
                leader_moves_freely_after_monitor
            ),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "operator_identity_recorded": False,
        },
        "next_authorized_stage": (
            "single_arm_readonly_preflight" if accepted else None
        ),
        "physical_output_commands": False,
        "receipt_writes_to_source_run": False,
    }


def build_single_arm_physical_receipt(
    *,
    source_run_id: str,
    gate_path: Path,
    source_hash_index_path: Path,
    source_done_path: Path,
    perception: str,
    no_cross_joint_instability: bool,
    leader_moves_freely_after_test: bool,
) -> dict[str, Any]:
    """Bind operator perception to the first five-joint physical stage."""

    if RUN_ID_PATTERN.fullmatch(source_run_id) is None:
        raise ValueError("source_run_id contains unsafe characters")
    if perception not in PERCEPTION_CHOICES:
        raise ValueError(f"unsupported perception verdict: {perception!r}")
    for path in (gate_path, source_hash_index_path, source_done_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_sha256 = sha256_file(gate_path)
    hashes = _hash_index_by_basename(source_hash_index_path)
    done = json.loads(source_done_path.read_text(encoding="utf-8"))
    checks = {
        "machine_gate_schema": (
            gate.get("schema_version")
            == "radeon_oneloop.haptic_arm_bench_gate.v1"
        ),
        "machine_gate_accepted": gate.get("accepted") is True,
        "machine_gate_stage": gate.get("stage") == "single_arm_physical",
        "machine_gate_physical_output_recorded": (
            gate.get("physical_output_commands") is True
        ),
        "machine_gate_waited_for_operator": (
            gate.get("operator_perception_gate")
            == "pending_separate_attestation"
        ),
        "gate_bound_by_source_hash_index": hashes.get(gate_path.name)
        == gate_sha256,
        "source_done_marker_accepted": (
            done.get("status") == "done_single_arm_physical_machine_accepted"
        ),
        "operator_perception_useful_comfortable": (
            perception == "useful_comfortable"
        ),
        "operator_reports_no_cross_joint_instability": (
            no_cross_joint_instability is True
        ),
        "operator_reports_free_motion_after_shutdown": (
            leader_moves_freely_after_test is True
        ),
    }
    accepted = all(checks.values())
    return {
        "schema_version": ARM_PHYSICAL_RECEIPT_SCHEMA_VERSION,
        "formal": False,
        "stage": "single_arm_physical",
        "selected_side": gate.get("selected_side"),
        "accepted": accepted,
        "checks": checks,
        "source": {
            "run_id": source_run_id,
            "gate_sha256": gate_sha256,
            "hash_index_sha256": sha256_file(source_hash_index_path),
            "done_marker_sha256": sha256_file(source_done_path),
        },
        "operator_attestation": {
            "perception": perception,
            "no_cross_joint_instability": no_cross_joint_instability,
            "leader_moves_freely_after_test": leader_moves_freely_after_test,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "operator_identity_recorded": False,
        },
        "next_authorized_stage": (
            "dual_arm_monitor_only" if accepted else None
        ),
        "physical_output_commands": False,
        "receipt_writes_to_source_run": False,
    }


def authorize_transition(receipt: dict[str, Any], *, target_stage: str) -> None:
    if receipt.get("schema_version") not in (
        SCHEMA_VERSION,
        MONITOR_RECEIPT_SCHEMA_VERSION,
        ARM_PHYSICAL_RECEIPT_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported haptic stage receipt schema")
    if receipt.get("accepted") is not True:
        raise ValueError("haptic stage receipt is not accepted")
    if receipt.get("next_authorized_stage") != target_stage:
        raise ValueError(
            f"receipt does not authorize target stage {target_stage!r}"
        )
    source = receipt.get("source") or {}
    for field in ("gate_sha256", "hash_index_sha256", "done_marker_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(source.get(field, ""))) is None:
            raise ValueError(f"receipt source field {field!r} is not a SHA-256")


def authorize_receipt_bundle(
    *,
    receipt_path: Path,
    hash_index_path: Path,
    done_path: Path,
    target_stage: str,
) -> dict[str, Any]:
    """Verify an immutable receipt run before a downstream stage starts."""

    for path in (receipt_path, hash_index_path, done_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authorize_transition(receipt, target_stage=target_stage)
    hashes = _hash_index_by_basename(hash_index_path)
    receipt_sha256 = sha256_file(receipt_path)
    if hashes.get(receipt_path.name) != receipt_sha256:
        raise ValueError("receipt is not bound by its run hash index")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    expected_done_status = {
        SCHEMA_VERSION: "done_single_joint_stage_accepted",
        MONITOR_RECEIPT_SCHEMA_VERSION: "done_single_arm_monitor_stage_accepted",
        ARM_PHYSICAL_RECEIPT_SCHEMA_VERSION: (
            "done_single_arm_physical_stage_accepted"
        ),
    }[receipt["schema_version"]]
    if done.get("status") != expected_done_status:
        raise ValueError("receipt run DONE marker has an unexpected status")
    return {
        "schema_version": "radeon_oneloop.haptic_stage_authorization.v1",
        "accepted": True,
        "target_stage": target_stage,
        "receipt_run_id": receipt_path.parent.name,
        "receipt_sha256": receipt_sha256,
        "receipt_hash_index_sha256": sha256_file(hash_index_path),
        "receipt_done_sha256": sha256_file(done_path),
        "physical_output_commands": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--source-hash-index", type=Path, required=True)
    parser.add_argument("--source-done", type=Path, required=True)
    parser.add_argument("--perception", choices=PERCEPTION_CHOICES, required=True)
    parser.add_argument("--leader-moves-freely-after-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_single_joint_receipt(
        source_run_id=args.source_run_id,
        gate_path=args.gate,
        source_hash_index_path=args.source_hash_index,
        source_done_path=args.source_done,
        perception=args.perception,
        leader_moves_freely_after_test=args.leader_moves_freely_after_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("haptic stage receipt was recorded but not accepted")


if __name__ == "__main__":
    main()
