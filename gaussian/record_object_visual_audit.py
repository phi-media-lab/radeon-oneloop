#!/usr/bin/env python3
"""Record an immutable, non-formal visual audit for object appearance runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


SCHEMA_VERSION = "radeon_oneloop.object_appearance_visual_audit.v1"
DONE_SCHEMA_VERSION = "radeon_oneloop.object_asset_stage_done.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash_manifest(root: Path) -> None:
    hash_path = root / "hashes.sha256"
    if not hash_path.is_file():
        raise ValueError(f"run is missing hashes.sha256: {root}")
    for line in hash_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError(f"hash entry escapes run directory: {relative}")
        if sha256_file(candidate) != expected:
            raise ValueError(f"hash mismatch: {candidate}")


def inspect_run(view: str, root: Path) -> dict[str, object]:
    root = root.resolve()
    if not (root / "DONE").is_file():
        raise ValueError(f"run is not complete: {root}")
    verify_hash_manifest(root)

    train_path = root / "train" / "train.json"
    manifest_path = root / "manifest.json"
    render_path = root / "train" / "val_00000.png"
    for required in (train_path, manifest_path, render_path):
        if not required.is_file():
            raise ValueError(f"run is missing required evidence: {required}")

    train = json.loads(train_path.read_text(encoding="utf-8"))
    val_images = train.get("val_images")
    if not isinstance(val_images, list) or len(val_images) != 1:
        raise ValueError(f"expected exactly one validation probe in {train_path}")
    val_image = val_images[0].get("image_path", "")
    expected_name = f"000_eval_probe_anchor_{view}.png"
    if Path(val_image).name != expected_name:
        raise ValueError(
            f"validation probe mismatch for {view}: expected {expected_name}, got {val_image}"
        )

    unique_train_names = sorted(
        {Path(item["image_path"]).name for item in train.get("train_images", [])}
    )
    expected_train_names = [
        "anchor_front.png",
        "anchor_left.png",
        "anchor_rear.png",
        "anchor_right.png",
    ]
    if unique_train_names != expected_train_names:
        raise ValueError(
            f"all-train probe must contain the four unique observed views: {unique_train_names}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "view": view,
        "run_path": str(root),
        "run_manifest_sha256": sha256_file(manifest_path),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "validation_probe": val_image,
        "validation_render_sha256": sha256_file(render_path),
        "train_image_names": unique_train_names,
    }


def write_atomic(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use VIEW=/absolute/run/path")
    view, raw_path = value.split("=", 1)
    if view not in {"front", "right", "rear", "left"}:
        raise argparse.ArgumentTypeError(f"unsupported view: {view}")
    return view, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument(
        "--review-status",
        default="accepted_observed_appearance_training_view_qa",
        choices=["accepted_observed_appearance_training_view_qa", "rejected"],
    )
    args = parser.parse_args()

    runs_by_view = dict(args.run)
    expected_views = {"front", "right", "rear", "left"}
    if set(runs_by_view) != expected_views or len(args.run) != 4:
        raise ValueError("exactly one run is required for front, right, rear, and left")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for reserved in (output / "audit_manifest.json", output / "hashes.sha256", output / "DONE"):
        if reserved.exists():
            raise FileExistsError(f"refusing to overwrite immutable audit evidence: {reserved}")

    montage_source = args.montage.resolve()
    if not montage_source.is_file():
        raise FileNotFoundError(montage_source)
    montage_target = output / montage_source.name
    if montage_source != montage_target:
        if montage_target.exists():
            raise FileExistsError(montage_target)
        shutil.copy2(montage_source, montage_target)

    runs = [inspect_run(view, runs_by_view[view]) for view in ("front", "right", "rear", "left")]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": args.asset_id,
        "stage": "observed_appearance_core",
        "formal": False,
        "review": {
            "status": args.review_status,
            "identity_consistent_across_views": args.review_status.startswith("accepted"),
            "orientation_consistent_across_views": args.review_status.startswith("accepted"),
            "known_limitations": [
                "hard-mask boundary halo",
                "black background outside the supervised object mask",
                "four-view sparse coverage is not sufficient for unseen-view claims",
            ],
        },
        "evaluation_protocol": {
            "kind": "duplicate_training_view_visual_probe",
            "held_out": False,
            "eligible_for_formal_metrics": False,
            "purpose": "direction, identity, and renderer integration QA only",
        },
        "montage": {
            "layout": "top=observed, bottom=render; columns=front,right,rear,left",
            "path": montage_target.name,
            "sha256": sha256_file(montage_target),
        },
        "runs": runs,
    }
    manifest_path = output / "audit_manifest.json"
    write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    hash_lines = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            hash_lines.append(f"{sha256_file(path)}  {path.name}\n")
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
