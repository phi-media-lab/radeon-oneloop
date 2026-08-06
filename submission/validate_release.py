#!/usr/bin/env python3
"""Fail closed when a competition release still contains placeholders."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


PLACEHOLDER = re.compile(r"\b(?:PENDING(?:_[A-Z_]+)?|TODO|INSERT FINAL)\b", re.IGNORECASE)


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def require_marker(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def reject_placeholders(path: Path, value: str | None = None) -> None:
    content = value if value is not None else require_file(path).read_text(encoding="utf-8")
    match = PLACEHOLDER.search(content)
    if match:
        raise ValueError(f"release placeholder {match.group(0)!r} remains in {path}")


def ffprobe(path: Path) -> dict[str, object]:
    value = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(require_file(path)),
        ],
        text=True,
    )
    return json.loads(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_restricted_artifacts(root: Path) -> int:
    policy_path = require_file(root / "submission/license_policy.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "radeon_oneloop.submission_license_policy.v1":
        raise ValueError("unexpected submission license policy schema")
    restricted = policy.get("restricted_artifacts")
    if not isinstance(restricted, list) or not restricted:
        raise ValueError("submission license policy has no restricted artifacts")
    forbidden_hashes: set[str] = set()
    for record in restricted:
        if record.get("competition_submission_clearance") is not False:
            raise ValueError("restricted model unexpectedly has submission clearance")
        hashes = record.get("known_descendant_sha256")
        if not isinstance(hashes, list) or not hashes:
            raise ValueError("restricted model lacks descendant hashes")
        forbidden_hashes.update(hashes)
    release_roots = [
        root / "artifacts/formal",
        root / "output/video",
        root / "output/pdf",
    ]
    checked = 0
    for release_root in release_roots:
        if not release_root.exists():
            continue
        for path in release_root.rglob("*"):
            if path.is_file():
                checked += 1
                if sha256_file(path) in forbidden_hashes:
                    raise ValueError(f"restricted noncommercial artifact entered release: {path}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    license_files_checked = validate_restricted_artifacts(root)

    sources = [
        root / "README.md",
        root / "reports/technical_report.md",
        root / "submission/official_pr_entry.md",
        root / "submission/video_script.md",
    ]
    for source in sources:
        reject_placeholders(source)

    report = require_file(root / "output/pdf/radeon-oneloop-technical-report.pdf")
    report_text = subprocess.check_output(["pdftotext", str(report), "-"], text=True)
    reject_placeholders(report, report_text)
    info = subprocess.check_output(["pdfinfo", str(report)], text=True)
    page_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if not page_match or int(page_match.group(1)) < 2:
        raise ValueError("technical report page count is invalid")

    video = require_file(root / "output/video/radeon-oneloop-demo.mp4")
    probe = ffprobe(video)
    seconds = float(probe["format"]["duration"])
    if not 180 <= seconds <= 300:
        raise ValueError(f"demo duration {seconds:.2f}s is outside 3-5 minutes")
    streams = probe["streams"]
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or videos[0].get("codec_name") != "h264":
        raise ValueError("demo must contain one H.264 video stream")
    if (videos[0].get("width"), videos[0].get("height")) != (1920, 1080):
        raise ValueError("demo must be 1920x1080")
    if len(audios) != 1 or audios[0].get("codec_name") != "aac":
        raise ValueError("demo must contain one AAC audio stream")

    evidence_labels = [
        "genesis_camera_corrected",
        "genesis_demo_video",
        "act_baseline",
        "act_phase_aware",
        "baseline_latency",
        "phase_latency",
        "baseline_reconstruction",
        "phase_reconstruction",
    ]
    for label in evidence_labels:
        evidence = root / "artifacts/formal" / label
        require_marker(evidence / "DONE")
        manifest = json.loads(require_file(evidence / "manifest.json").read_text())
        if manifest.get("status") != "done" or manifest.get("formal") is not True:
            raise ValueError(f"formal evidence is not done: {evidence}")
        if manifest.get("host") != "radeon-c" or manifest.get("gpu_uid") != "0x153f7d55778ab659":
            raise ValueError(f"formal identity mismatch: {evidence}")

    print(
        json.dumps(
            {
                "status": "passed",
                "report_pages": int(page_match.group(1)),
                "video_seconds": seconds,
                "formal_evidence_directories": len(evidence_labels),
                "release_files_checked_against_license_policy": license_files_checked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
