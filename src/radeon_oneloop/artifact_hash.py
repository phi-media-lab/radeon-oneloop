"""Deterministic SHA-256 fingerprint for a checkpoint or evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, list[dict[str, str | int]]]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    records: list[dict[str, str | int]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        item_hash = file_sha256(path)
        digest.update(f"F\0{relative}\0{size}\0{item_hash}\n".encode())
        records.append({"path": relative, "bytes": size, "sha256": item_hash})
    if not records:
        raise ValueError(f"artifact tree is empty: {root}")
    return digest.hexdigest(), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    fingerprint, records = tree_sha256(args.path)
    result = {
        "schema_version": "radeon_oneloop.artifact_tree.v1",
        "root": str(args.path.resolve()),
        "tree_sha256": fingerprint,
        "files": records,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.ledger:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
