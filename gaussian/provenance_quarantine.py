#!/usr/bin/env python3
"""Reject artifacts whose lineage contains a visually invalid real2sim prior."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable


SCHEMA_VERSION = "radeon_oneloop.real2sim_quarantine.v1"
DEFAULT_REGISTRY = Path(__file__).with_name("real2sim_quarantine.json")


class QuarantinedLineageError(ValueError):
    """Raised before a quarantined artifact can enter a new mainline stage."""


@dataclass(frozen=True)
class QuarantineMatch:
    entry_id: str
    token: str
    token_kind: str
    source_label: str


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = _load_object(path, "quarantine registry")
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected real2sim quarantine registry schema")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("quarantine registry must contain entries")
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("decision") != "rejected":
            raise ValueError("every quarantine entry must be an explicit rejection")
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id or entry_id in seen_ids:
            raise ValueError("quarantine entry ids must be unique non-empty strings")
        seen_ids.add(entry_id)
        tokens = [
            *entry.get("run_ids", []),
            *entry.get("sha256", []),
            *entry.get("dataset_hashes", []),
        ]
        if not tokens or not all(isinstance(item, str) and item for item in tokens):
            raise ValueError(f"quarantine entry {entry_id!r} has invalid trigger tokens")
    return registry


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)


def find_quarantine_matches(
    sources: Iterable[tuple[str, Any]], *, registry_path: Path = DEFAULT_REGISTRY
) -> list[QuarantineMatch]:
    registry = load_registry(registry_path)
    trigger_index: dict[str, tuple[str, str]] = {}
    for entry in registry["entries"]:
        entry_id = str(entry["id"])
        for kind, field in (
            ("run_id", "run_ids"),
            ("sha256", "sha256"),
            ("dataset_hash", "dataset_hashes"),
        ):
            for token in entry.get(field, []):
                previous = trigger_index.get(token)
                if previous is not None and previous != (entry_id, kind):
                    raise ValueError(f"ambiguous quarantine trigger: {token}")
                trigger_index[token] = (entry_id, kind)

    matches: set[QuarantineMatch] = set()
    for label, payload in sources:
        for token in _string_values(payload):
            hit = trigger_index.get(token)
            if hit is not None:
                matches.add(
                    QuarantineMatch(
                        entry_id=hit[0], token=token, token_kind=hit[1], source_label=label
                    )
                )
    return sorted(
        matches,
        key=lambda item: (item.entry_id, item.source_label, item.token_kind, item.token),
    )


def assert_not_quarantined(
    sources: Iterable[tuple[str, Any]], *, registry_path: Path = DEFAULT_REGISTRY
) -> None:
    matches = find_quarantine_matches(sources, registry_path=registry_path)
    if not matches:
        return
    summary = "; ".join(
        f"{match.entry_id}:{match.token_kind}={match.token} in {match.source_label}"
        for match in matches
    )
    raise QuarantinedLineageError(
        "rejected real2sim lineage may only remain a negative control: " + summary
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--check-json", action="append", type=Path, default=[])
    parser.add_argument("--check-value", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sources: list[tuple[str, Any]] = [
        (str(path), _load_object(path, "lineage manifest")) for path in args.check_json
    ]
    sources.extend(("command_line_value", value) for value in args.check_value)
    if not sources:
        raise SystemExit("at least one --check-json or --check-value is required")
    try:
        assert_not_quarantined(sources, registry_path=args.registry)
    except QuarantinedLineageError as exc:
        print(f"quarantine gate rejected lineage: {exc}", file=sys.stderr)
        raise SystemExit(65) from None
    print(
        json.dumps(
            {
                "schema_version": "radeon_oneloop.real2sim_quarantine_check.v1",
                "status": "passed",
                "checked_sources": [label for label, _ in sources],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
