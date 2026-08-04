"""Validate and summarize nonformal AMD Gaussian development evidence.

The public bundle deliberately excludes raw leader samples.  The collector
downloads those samples only into an OS temporary directory, derives bounded
range/physical-output facts, and publishes this summary instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "radeon_oneloop.development_evidence_summary.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _final_json_with_schema(path: Path, schema: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    for start in reversed([index for index, character in enumerate(text) if character == "{"]):
        try:
            value = json.loads(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == schema:
            return value
    raise ValueError(f"final {schema} document is missing from {path}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_hashes(root: Path, relative_paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        _require(path.is_file(), f"required evidence file is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def summarize_orbit(
    root: Path, *, expected_ply_sha256: str, visual_review: str
) -> dict[str, Any]:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    metrics = _json(root / "artifacts/metrics.json")
    _require(isinstance(manifest, dict), "orbit manifest must be a mapping")
    _require(manifest.get("formal") is False, "AMD orbit execution must be nonformal")
    _require(
        manifest.get("host_role") == "amd_apu_nonformal_visual_audit",
        "unexpected orbit host role",
    )
    _require(manifest.get("physical_output") is False, "orbit declares physical output")
    _require(metrics.get("accepted_numeric") is True, "orbit numeric gate is not accepted")
    _require(metrics.get("physical_output") is False, "orbit metrics declare physical output")
    _require(metrics.get("formal") is False, "AMD orbit metrics must be nonformal")
    asset = metrics.get("asset") or {}
    _require(asset.get("formal") is True, "orbit does not consume a formal upstream asset")
    _require(
        (asset.get("hashes") or {}).get("ply") == expected_ply_sha256,
        "orbit PLY does not match the expected formal asset",
    )
    orbit = metrics.get("orbit") or {}
    _require(orbit.get("cycle_closure_rgb_mae") == 0.0, "orbit does not close exactly")
    _require(orbit.get("border_contact_frames") == 0, "orbit touches the image border")
    _require(bool(visual_review.strip()), "a human visual-review label is required")
    paths = [
        "DONE",
        "manifest.yaml",
        "hashes.sha256",
        "artifacts/metrics.json",
        "artifacts/orbit_contact_sheet.png",
        "artifacts/orbit_360.mp4",
    ]
    return {
        "schema_version": SCHEMA,
        "mode": "gaussian_orbit_audit",
        "formal": False,
        "accepted": True,
        "source_run_id": root.name,
        "asset": asset,
        "numeric": {
            "frames_without_duplicate_endpoint": orbit.get(
                "frames_without_duplicate_endpoint"
            ),
            "cycle_closure_rgb_mae": orbit.get("cycle_closure_rgb_mae"),
            "border_contact_frames": orbit.get("border_contact_frames"),
            "alpha_support_fraction": orbit.get("alpha_support_fraction"),
            "render_ms": metrics.get("render_ms"),
        },
        "visual_review": visual_review,
        "physical_output": False,
        "heldout_quality_claim": False,
        "task_success_evaluated": False,
        "source_files_sha256": _source_hashes(root, paths),
    }


def summarize_live(root: Path, *, expected_ply_sha256: str) -> dict[str, Any]:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    gate = _json(root / "gate.json")
    consumer = _json(root / "consumer/metrics.json")
    renderer = _json(root / "renderer/metrics.json")
    publisher = _final_json_with_schema(
        root / "publisher.log", "radeon_oneloop.leader_publisher.v1"
    )
    _require(isinstance(manifest, dict), "live manifest must be a mapping")
    _require(manifest.get("formal") is False, "AMD live execution must be nonformal")
    _require(
        manifest.get("host_role") == "amd_apu_nonformal_runtime_integration",
        "unexpected live host role",
    )
    _require(manifest.get("physical_output") is False, "live manifest declares output")
    _require(gate.get("accepted") is True, "live gate is not accepted")
    _require(gate.get("physical_output") is False, "live gate declares physical output")
    _require(consumer.get("physical_output_commands") is False, "control emitted output")
    _require(publisher.get("physical_output_commands") is False, "publisher emitted output")
    haptic = publisher.get("haptic_feedback") or {}
    _require(haptic.get("output_commands") == 0, "publisher reports motor commands")
    _require(renderer.get("accepted") is True, "renderer gate is not accepted")
    _require(renderer.get("physical_output") is False, "renderer declares output")
    asset = renderer.get("asset") or {}
    _require(asset.get("formal") is True, "live run does not consume a formal asset")
    _require(
        (asset.get("hashes") or {}).get("ply") == expected_ply_sha256,
        "live PLY does not match the expected formal asset",
    )
    action_range = publisher.get("action_range") or {}
    spans = action_range.get("span")
    _require(
        isinstance(spans, list) and len(spans) == 12,
        "publisher does not contain the frozen 12-channel action range",
    )
    paths = [
        "DONE",
        "manifest.yaml",
        "hashes.sha256",
        "gate.json",
        "consumer/metrics.json",
        "renderer/READY",
        "renderer/metrics.json",
        "renderer/live_gaussian_first.png",
        "renderer/live_gaussian_final.png",
        "renderer/live_gaussian.mp4",
        "publisher.log",
    ]
    return {
        "schema_version": SCHEMA,
        "mode": "decoupled_gaussian_live_gate",
        "formal": False,
        "accepted": True,
        "source_run_id": root.name,
        "asset": asset,
        "gate": gate,
        "control": {
            "packets": consumer.get("packets"),
            "watchdog": consumer.get("watchdog"),
            "physical_output_commands": False,
        },
        "publisher": {
            "action_names": action_range.get("action_names"),
            "span": spans,
            "samples": action_range.get("samples"),
            "capture_start_gated": action_range.get("capture_start_gated"),
            "capture_started": action_range.get("capture_started"),
            "haptic_output_commands": 0,
            "physical_output_commands": False,
        },
        "renderer": {
            "appearance": renderer.get("appearance"),
            "render": renderer.get("render"),
        },
        "task_success_evaluated": False,
        "source_files_sha256": _source_hashes(root, paths),
    }


def summarize(
    root: Path,
    *,
    mode: str,
    expected_ply_sha256: str,
    visual_review: str = "",
) -> dict[str, Any]:
    root = root.resolve()
    _require((root / "DONE").is_file(), "source run has no DONE marker")
    _require(not (root / "FAILED").exists(), "source run has a FAILED marker")
    _require(
        len(expected_ply_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_ply_sha256),
        "expected PLY SHA-256 is invalid",
    )
    if mode == "orbit":
        return summarize_orbit(
            root,
            expected_ply_sha256=expected_ply_sha256,
            visual_review=visual_review,
        )
    if mode == "live":
        return summarize_live(root, expected_ply_sha256=expected_ply_sha256)
    raise ValueError(f"unsupported development evidence mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("orbit", "live"), required=True)
    parser.add_argument("--expected-ply-sha256", required=True)
    parser.add_argument("--visual-review", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.source,
        mode=args.mode,
        expected_ply_sha256=args.expected_ply_sha256,
        visual_review=args.visual_review,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
