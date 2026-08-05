#!/usr/bin/env python3
"""Build and seal the human review boundary for the live full-geometry asset.

The packet joins four separate, content-addressed facts without promoting any
generated view to observed or metric evidence:

* the accepted low-confidence SEVA pseudo-view review;
* the variable-geometry 360-degree Gaussian orbit;
* the hardware-isolated, pose-synchronous Genesis live render; and
* the exact PLY/camera/provenance files used by those renders.

Packet construction never records human acceptance.  A separate ``seal``
operation is intentionally required after the project owner reviews the
packet.  The resulting receipt authorizes only read-only dual-leader monitor
mode, never force feedback or physical follower output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping


PACKET_SCHEMA = "radeon_oneloop.full_geometry_visual_review_packet.v1"
PACKET_DONE_SCHEMA = "radeon_oneloop.full_geometry_visual_review_packet_done.v1"
RECEIPT_SCHEMA = "radeon_oneloop.full_geometry_visual_review_receipt.v1"
RECEIPT_DONE_SCHEMA = "radeon_oneloop.full_geometry_visual_review_receipt_done.v1"
NEXT_STAGE = "dual_leader_monitor_only"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class FullGeometryVisualReviewError(ValueError):
    """Raised when evidence or a visual authorization bundle fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullGeometryVisualReviewError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FullGeometryVisualReviewError(f"{label} must be a JSON object")
    return value


def _required_file(path: Path) -> Path:
    value = path.resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _read_hash_index(path: Path) -> dict[str, str]:
    """Read a GNU-style hash index keyed by normalized relative path."""

    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or HEX_SHA256.fullmatch(fields[0]) is None:
            raise FullGeometryVisualReviewError(
                f"invalid hash-index line {line_number}: {path}"
            )
        raw = fields[1].lstrip("*")
        while raw.startswith("./"):
            raw = raw[2:]
        if not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise FullGeometryVisualReviewError(f"unsafe hash-index path: {raw!r}")
        if raw in result:
            raise FullGeometryVisualReviewError(f"duplicate hash-index path: {raw}")
        result[raw] = fields[0]
    if not result:
        raise FullGeometryVisualReviewError(f"empty hash index: {path}")
    return result


def _require_index_binding(
    index: Mapping[str, str], relative: str, actual_path: Path
) -> None:
    actual = sha256_file(actual_path)
    if index.get(relative) != actual:
        raise FullGeometryVisualReviewError(
            f"hash index does not bind {relative}: expected {index.get(relative)!r}, "
            f"actual {actual}"
        )


def _copy(source: Path, destination: Path) -> dict[str, Any]:
    _required_file(source)
    shutil.copyfile(source, destination)
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def _asset_contract(asset_root: Path) -> dict[str, Any]:
    ply = _required_file(asset_root / "appearance_full_geometry_canonical.ply")
    cameras = _required_file(asset_root / "cameras_full_geometry.json")
    provenance_path = _required_file(asset_root / "provenance.json")
    provenance = _load_json(provenance_path, "asset provenance")
    if provenance.get("schema_version") != "radeon_oneloop.gaussian_canonicalization.v2":
        raise FullGeometryVisualReviewError("unexpected canonical asset provenance schema")
    if provenance.get("provenance_class") != "generated_full_geometry_candidate":
        raise FullGeometryVisualReviewError("asset is not the full-geometry candidate")
    if provenance.get("formal") is not False:
        raise FullGeometryVisualReviewError("generated geometry must remain nonformal")
    if provenance.get("eligible_for_heldout_real_metrics") is not False:
        raise FullGeometryVisualReviewError("generated geometry entered held-out metrics")
    hashes = {
        "ply": sha256_file(ply),
        "cameras": sha256_file(cameras),
        "provenance": sha256_file(provenance_path),
    }
    if provenance.get("output_ply_sha256") != hashes["ply"]:
        raise FullGeometryVisualReviewError("provenance does not bind the canonical PLY")
    return {
        "hashes": hashes,
        "gaussian_count": provenance.get("gaussian_count"),
        "provenance_class": provenance["provenance_class"],
        "formal": False,
        "collision_eligible": False,
        "heldout_real_metrics_eligible": False,
        "training_lineage": {
            "run_id": provenance.get("training_lineage", {}).get("training_run_id"),
            "dataset_manifest_sha256": provenance.get("training_lineage", {}).get(
                "dataset_manifest_sha256"
            ),
            "vksplat_commit": provenance.get("training_lineage", {}).get(
                "vksplat_commit"
            ),
        },
    }


def _validate_source_review(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    review_path = _required_file(root / "accepted_review.json")
    metrics_path = _required_file(root / "metrics.json")
    anchor_path = _required_file(root / "real_generated_difference_anchors.png")
    orbit_path = _required_file(root / "samples-rgb.mp4")
    review = _load_json(review_path, "SEVA pseudo-view human review")
    metrics = _load_json(metrics_path, "SEVA pseudo-view metrics")
    if review.get("schema_version") != "radeon_oneloop.seva_four_view_orbit_review.v1":
        raise FullGeometryVisualReviewError("unexpected SEVA review schema")
    if review.get("decision") != "accepted_low_confidence_pseudoviews":
        raise FullGeometryVisualReviewError("SEVA pseudo-views were not accepted")
    if review.get("accepted_role") != "generated_low_confidence_appearance_pseudoviews":
        raise FullGeometryVisualReviewError("SEVA review granted an unexpected role")
    if metrics.get("frames") != 49 or metrics.get("formal") is not False:
        raise FullGeometryVisualReviewError("SEVA review metrics violate the 49-view boundary")
    expected_anchor_sha = review.get("evidence", {}).get("anchor_comparison_sha256")
    if expected_anchor_sha != sha256_file(anchor_path):
        raise FullGeometryVisualReviewError("SEVA review does not bind the anchor comparison")
    return review, {
        "review": review_path,
        "metrics": metrics_path,
        "anchors": anchor_path,
        "orbit": orbit_path,
    }


def _validate_orbit_run(
    root: Path, asset: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Path]]:
    metrics_path = _required_file(root / "artifacts/metrics.json")
    contact_path = _required_file(root / "artifacts/orbit_contact_sheet.png")
    video_path = _required_file(root / "artifacts/orbit_360.mp4")
    done_path = _required_file(root / "DONE")
    index_path = _required_file(root / "hashes.sha256")
    metrics = _load_json(metrics_path, "full-geometry orbit metrics")
    index = _read_hash_index(index_path)
    for relative, path in (
        ("artifacts/metrics.json", metrics_path),
        ("artifacts/orbit_contact_sheet.png", contact_path),
        ("artifacts/orbit_360.mp4", video_path),
    ):
        _require_index_binding(index, relative, path)
    if metrics.get("accepted_numeric") is not True or metrics.get("formal") is not False:
        raise FullGeometryVisualReviewError("full-geometry orbit gate is not accepted")
    frames = metrics.get("frames")
    if not isinstance(frames, list) or len(frames) < 72:
        raise FullGeometryVisualReviewError("full-geometry orbit lacks a complete turn")
    if metrics.get("eligible_for_heldout_real_metrics") is not False:
        raise FullGeometryVisualReviewError("orbit was incorrectly used as held-out evidence")
    if metrics.get("asset", {}).get("hashes") != asset["hashes"]:
        raise FullGeometryVisualReviewError("orbit run used a different asset")
    return metrics, {
        "metrics": metrics_path,
        "contact": contact_path,
        "video": video_path,
        "done": done_path,
        "hashes": index_path,
    }


def _validate_live_run(
    root: Path, asset: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Path]]:
    gate_path = _required_file(root / "gate.json")
    renderer_metrics_path = _required_file(root / "renderer/metrics.json")
    video_path = _required_file(root / "live_gaussian.mp4")
    first_path = _required_file(root / "live_gaussian_first.png")
    final_path = _required_file(root / "live_gaussian_final.png")
    index_path = _required_file(root / "hashes.sha256")
    gate = _load_json(gate_path, "authoritative synthetic gate")
    renderer = _load_json(renderer_metrics_path, "live Gaussian renderer metrics")
    index = _read_hash_index(index_path)
    _require_index_binding(index, "gate.json", gate_path)
    _require_index_binding(index, "renderer/metrics.json", renderer_metrics_path)
    # Local evidence collection flattens these three renderer artifacts while
    # preserving their bytes and the original remote hash-index names.
    for relative, path in (
        ("renderer/live_gaussian.mp4", video_path),
        ("renderer/live_gaussian_first.png", first_path),
        ("renderer/live_gaussian_final.png", final_path),
    ):
        _require_index_binding(index, relative, path)
    if gate.get("schema_version") != "radeon_oneloop.gaussian_authoritative_synthetic_gate.v1":
        raise FullGeometryVisualReviewError("unexpected live gate schema")
    if gate.get("accepted") is not True or not all(gate.get("checks", {}).values()):
        raise FullGeometryVisualReviewError("hardware-isolated live gate is not accepted")
    if gate.get("physical_output") is not False or gate.get("serial_or_usb_access") is not False:
        raise FullGeometryVisualReviewError("visual preflight touched physical hardware")
    appearance = gate.get("renderer_appearance", {})
    if (
        appearance.get("object_visualization") is not False
        or appearance.get("compositor") != "gaussian_self_depth"
        or not str(appearance.get("object_mesh_path", "")).endswith("_collision.obj")
    ):
        raise FullGeometryVisualReviewError("legacy visible OBJ remains in the live render")
    if renderer.get("asset", {}).get("hashes") != asset["hashes"]:
        raise FullGeometryVisualReviewError("live run used a different asset")
    if renderer.get("full_geometry_candidate") is not True:
        raise FullGeometryVisualReviewError("live renderer did not select full geometry")
    return gate, {
        "gate": gate_path,
        "renderer_metrics": renderer_metrics_path,
        "video": video_path,
        "first": first_path,
        "final": final_path,
        "hashes": index_path,
    }


def _write_hash_index(root: Path, *, excluded: set[str]) -> str:
    lines = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in excluded:
            lines.append(f"{sha256_file(path)}  {path.name}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def build_packet(
    *,
    source_review_root: Path,
    orbit_run_root: Path,
    live_run_root: Path,
    asset_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Create one immutable, pending-human-review evidence packet."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    asset = _asset_contract(asset_root.resolve())
    source_review, source_files = _validate_source_review(source_review_root.resolve())
    orbit_metrics, orbit_files = _validate_orbit_run(orbit_run_root.resolve(), asset)
    live_gate, live_files = _validate_live_run(live_run_root.resolve(), asset)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        evidence = {
            "real_vs_generated_anchors": _copy(
                source_files["anchors"], staging / "01_real_vs_generated_anchors.png"
            ),
            "generated_49_view_orbit": _copy(
                source_files["orbit"], staging / "02_generated_49_view_orbit.mp4"
            ),
            "full_geometry_orbit_contact_sheet": _copy(
                orbit_files["contact"], staging / "03_full_geometry_orbit_contact_sheet.png"
            ),
            "full_geometry_orbit_video": _copy(
                orbit_files["video"], staging / "04_full_geometry_orbit_360.mp4"
            ),
            "live_first_frame": _copy(
                live_files["first"], staging / "05_live_first.png"
            ),
            "live_final_frame": _copy(
                live_files["final"], staging / "06_live_final.png"
            ),
            "live_video": _copy(live_files["video"], staging / "07_live_gaussian.mp4"),
            "asset_provenance": _copy(
                asset_root.resolve() / "provenance.json", staging / "asset_provenance.json"
            ),
            "asset_cameras": _copy(
                asset_root.resolve() / "cameras_full_geometry.json",
                staging / "asset_cameras.json",
            ),
            "source_review": _copy(
                source_files["review"], staging / "source_pseudoview_review.json"
            ),
            "orbit_metrics": _copy(
                orbit_files["metrics"], staging / "full_geometry_orbit_metrics.json"
            ),
            "live_gate": _copy(live_files["gate"], staging / "live_gate.json"),
            "live_renderer_metrics": _copy(
                live_files["renderer_metrics"], staging / "live_renderer_metrics.json"
            ),
        }
        packet = {
            "schema_version": PACKET_SCHEMA,
            "created_utc": utc_now(),
            "status": "pending_project_owner_visual_review",
            "formal": False,
            "asset": asset,
            "source_pseudoview_review": {
                "decision": source_review["decision"],
                "accepted_role": source_review["accepted_role"],
                "review_sha256": sha256_file(source_files["review"]),
                "geometry_role_accepted": False,
            },
            "orbit_gate": {
                "accepted_numeric": orbit_metrics["accepted_numeric"],
                "rendered_frame_records": len(orbit_metrics["frames"]),
                "heldout_real_metrics_eligible": False,
            },
            "live_gate": {
                "accepted": live_gate["accepted"],
                "physical_output": False,
                "serial_or_usb_access": False,
                "object_visualization": live_gate["renderer_appearance"][
                    "object_visualization"
                ],
                "compositor": live_gate["renderer_appearance"]["compositor"],
            },
            "evidence": evidence,
            "required_human_checks": {
                "correct_object_identity_and_front_back_orientation": None,
                "complete_orbit_geometry_acceptable_for_visual_demo": None,
                "legacy_procedural_obj_not_visible": None,
                "live_scale_pose_and_occlusion_acceptable": None,
            },
            "claim_boundary": [
                "The 49 SEVA views and resulting geometry are generated hypotheses.",
                "Acceptance authorizes nonformal visual demonstration only.",
                "Acceptance does not establish metric geometry, collision fidelity, or held-out-real quality.",
                "The collision OBJ remains an invisible approximate physics proxy.",
                "A sealed receipt can authorize only monitor-only dual-leader input with no physical output.",
            ],
        }
        packet_path = staging / "review_packet.json"
        packet_path.write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "REVIEW.md").write_text(
            "# Full-geometry visual acceptance packet\n\n"
            "Status: **pending project-owner review**.\n\n"
            "Review in order:\n\n"
            "1. `01_real_vs_generated_anchors.png`: the top row is the four real product views; the middle row is SEVA at the same anchors.\n"
            "2. `02_generated_49_view_orbit.mp4`: generated camera-controlled conditioning orbit.\n"
            "3. `03_full_geometry_orbit_contact_sheet.png` and `04_full_geometry_orbit_360.mp4`: the variable-geometry Gaussian itself.\n"
            "4. `05_live_first.png`, `06_live_final.png`, and `07_live_gaussian.mp4`: pose-synchronous Genesis integration.\n\n"
            "Acceptance requires correct identity/orientation, an acceptable complete orbit, no visible legacy OBJ, and acceptable live registration/occlusion. It does not assert metric or collision fidelity.\n",
            encoding="utf-8",
        )
        hashes_sha = _write_hash_index(
            staging, excluded={"hashes.sha256", "DONE", "FAILED"}
        )
        done = {
            "schema_version": PACKET_DONE_SCHEMA,
            "status": "done_packet_pending_project_owner_review",
            "review_packet_sha256": sha256_file(packet_path),
            "hashes_sha256": hashes_sha,
            "completed_utc": utc_now(),
        }
        (staging / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return packet
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        raise


def _validate_packet_bundle(packet_dir: Path) -> dict[str, Any]:
    root = packet_dir.resolve()
    packet_path = _required_file(root / "review_packet.json")
    hashes_path = _required_file(root / "hashes.sha256")
    done_path = _required_file(root / "DONE")
    packet = _load_json(packet_path, "visual review packet")
    done = _load_json(done_path, "visual review packet DONE")
    index = _read_hash_index(hashes_path)
    for relative, expected in index.items():
        path = _required_file(root / relative)
        if sha256_file(path) != expected:
            raise FullGeometryVisualReviewError(f"packet file hash mismatch: {relative}")
    if packet.get("schema_version") != PACKET_SCHEMA:
        raise FullGeometryVisualReviewError("unexpected visual review packet schema")
    if packet.get("status") != "pending_project_owner_visual_review":
        raise FullGeometryVisualReviewError("packet is not pending owner review")
    if done.get("schema_version") != PACKET_DONE_SCHEMA:
        raise FullGeometryVisualReviewError("unexpected packet DONE schema")
    if done.get("review_packet_sha256") != sha256_file(packet_path):
        raise FullGeometryVisualReviewError("packet DONE does not bind its manifest")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise FullGeometryVisualReviewError("packet DONE does not bind its hash index")
    return packet


def seal_receipt(
    *,
    packet_dir: Path,
    output: Path,
    decision: str,
    correct_identity_orientation: bool,
    complete_orbit_acceptable: bool,
    legacy_obj_not_visible: bool,
    live_registration_acceptable: bool,
) -> dict[str, Any]:
    """Record a separate human decision without modifying its source packet."""

    packet_root = packet_dir.resolve()
    packet = _validate_packet_bundle(packet_root)
    if decision not in {"accepted", "rejected"}:
        raise FullGeometryVisualReviewError("decision must be accepted or rejected")
    checks = {
        "correct_object_identity_and_front_back_orientation": bool(
            correct_identity_orientation
        ),
        "complete_orbit_geometry_acceptable_for_visual_demo": bool(
            complete_orbit_acceptable
        ),
        "legacy_procedural_obj_not_visible": bool(legacy_obj_not_visible),
        "live_scale_pose_and_occlusion_acceptable": bool(
            live_registration_acceptable
        ),
    }
    accepted = decision == "accepted" and all(checks.values())
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "created_utc": utc_now(),
            "formal": False,
            "accepted": accepted,
            "decision": decision,
            "reviewer_role": "project_owner_human_review",
            "checks": checks,
            "source_packet": {
                "review_packet_sha256": sha256_file(
                    packet_root / "review_packet.json"
                ),
                "hash_index_sha256": sha256_file(packet_root / "hashes.sha256"),
                "done_sha256": sha256_file(packet_root / "DONE"),
            },
            "asset_hashes": packet["asset"]["hashes"],
            "next_authorized_stage": NEXT_STAGE if accepted else None,
            "physical_output_commands": False,
            "haptic_output_authorized": False,
            "receipt_writes_to_source_packet": False,
            "claim_boundary": packet["claim_boundary"],
        }
        receipt_path = staging / "receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hash_index(
            staging, excluded={"hashes.sha256", "DONE", "FAILED"}
        )
        done = {
            "schema_version": RECEIPT_DONE_SCHEMA,
            "status": (
                "done_visual_review_accepted"
                if accepted
                else "done_visual_review_rejected"
            ),
            "receipt_sha256": sha256_file(receipt_path),
            "hashes_sha256": hashes_sha,
            "completed_utc": utc_now(),
        }
        (staging / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return receipt
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        raise


def authorize_receipt(
    *, receipt_dir: Path, expected_asset_ply_sha256: str, target_stage: str
) -> dict[str, Any]:
    """Fail closed unless one sealed receipt authorizes the exact live asset."""

    if HEX_SHA256.fullmatch(expected_asset_ply_sha256) is None:
        raise FullGeometryVisualReviewError("expected PLY hash is invalid")
    root = receipt_dir.resolve()
    receipt_path = _required_file(root / "receipt.json")
    hashes_path = _required_file(root / "hashes.sha256")
    done_path = _required_file(root / "DONE")
    receipt = _load_json(receipt_path, "visual acceptance receipt")
    done = _load_json(done_path, "visual acceptance receipt DONE")
    index = _read_hash_index(hashes_path)
    _require_index_binding(index, "receipt.json", receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise FullGeometryVisualReviewError("unexpected visual receipt schema")
    if receipt.get("accepted") is not True or receipt.get("decision") != "accepted":
        raise FullGeometryVisualReviewError("visual receipt is not accepted")
    if not all(receipt.get("checks", {}).values()):
        raise FullGeometryVisualReviewError("visual receipt has a failed human check")
    if receipt.get("next_authorized_stage") != target_stage or target_stage != NEXT_STAGE:
        raise FullGeometryVisualReviewError(
            f"visual receipt does not authorize {target_stage!r}"
        )
    if receipt.get("asset_hashes", {}).get("ply") != expected_asset_ply_sha256:
        raise FullGeometryVisualReviewError("visual receipt binds a different PLY")
    if receipt.get("physical_output_commands") is not False:
        raise FullGeometryVisualReviewError("visual receipt cannot authorize physical output")
    if receipt.get("haptic_output_authorized") is not False:
        raise FullGeometryVisualReviewError("visual receipt cannot authorize haptics")
    if done.get("schema_version") != RECEIPT_DONE_SCHEMA:
        raise FullGeometryVisualReviewError("unexpected visual receipt DONE schema")
    if done.get("status") != "done_visual_review_accepted":
        raise FullGeometryVisualReviewError("visual receipt DONE is not accepted")
    if done.get("receipt_sha256") != sha256_file(receipt_path):
        raise FullGeometryVisualReviewError("receipt DONE does not bind the receipt")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise FullGeometryVisualReviewError("receipt DONE does not bind the hash index")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-review-root", type=Path, required=True)
    build.add_argument("--orbit-run-root", type=Path, required=True)
    build.add_argument("--live-run-root", type=Path, required=True)
    build.add_argument("--asset-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--packet-dir", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--decision", choices=("accepted", "rejected"), required=True)
    seal.add_argument("--correct-identity-orientation", action="store_true")
    seal.add_argument("--complete-orbit-acceptable", action="store_true")
    seal.add_argument("--legacy-obj-not-visible", action="store_true")
    seal.add_argument("--live-registration-acceptable", action="store_true")

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--receipt-dir", type=Path, required=True)
    authorize.add_argument("--expected-asset-ply-sha256", required=True)
    authorize.add_argument("--target-stage", choices=(NEXT_STAGE,), required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_packet(
            source_review_root=args.source_review_root,
            orbit_run_root=args.orbit_run_root,
            live_run_root=args.live_run_root,
            asset_root=args.asset_root,
            output=args.output,
        )
    elif args.command == "seal":
        result = seal_receipt(
            packet_dir=args.packet_dir,
            output=args.output,
            decision=args.decision,
            correct_identity_orientation=args.correct_identity_orientation,
            complete_orbit_acceptable=args.complete_orbit_acceptable,
            legacy_obj_not_visible=args.legacy_obj_not_visible,
            live_registration_acceptable=args.live_registration_acceptable,
        )
    else:
        result = authorize_receipt(
            receipt_dir=args.receipt_dir,
            expected_asset_ply_sha256=args.expected_asset_ply_sha256,
            target_stage=args.target_stage,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
