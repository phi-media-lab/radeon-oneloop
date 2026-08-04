#!/usr/bin/env python3
"""Split and reassemble a hashed artifact for parallel low-bandwidth transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_artifact(source: Path, output: Path, chunk_bytes: int) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    output.mkdir(parents=True)
    parts = []
    with source.open("rb") as stream:
        index = 0
        while True:
            payload = stream.read(chunk_bytes)
            if not payload:
                break
            part = output / f"part-{index:05d}.bin"
            part.write_bytes(payload)
            parts.append(
                {"relpath": part.name, "bytes": len(payload), "sha256": sha256_file(part)}
            )
            index += 1
    manifest = {
        "schema_version": "radeon_oneloop.chunked_artifact.v1",
        "source_name": source.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "chunk_bytes": chunk_bytes,
        "parts": parts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def join_artifact(manifest_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    temporary = output.with_name(f".{output.name}.tmp")
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("xb") as target:
            for item in manifest["parts"]:
                part = (manifest_path.parent / item["relpath"]).resolve()
                if not part.is_relative_to(manifest_path.parent.resolve()):
                    raise ValueError(f"part escapes chunk directory: {item['relpath']}")
                if part.stat().st_size != item["bytes"] or sha256_file(part) != item["sha256"]:
                    raise ValueError(f"chunk hash or size mismatch: {part}")
                with part.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        target.write(block)
                        digest.update(block)
                        total += len(block)
        if total != manifest["source_bytes"] or digest.hexdigest() != manifest["source_sha256"]:
            raise ValueError("reassembled artifact hash or size mismatch")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": manifest["schema_version"],
        "output": str(output),
        "bytes": total,
        "sha256": digest.hexdigest(),
        "part_count": len(manifest["parts"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--source", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    join = subparsers.add_parser("join")
    join.add_argument("--manifest", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "split":
        result = split_artifact(args.source.resolve(), args.output.resolve(), args.chunk_bytes)
    else:
        result = join_artifact(args.manifest.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
