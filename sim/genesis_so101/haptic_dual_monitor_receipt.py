#!/usr/bin/env python3
"""Seal the operator judgment for a dual-arm monitor-only stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .haptic_stage_receipt import (
    DUAL_MAPPING_CHOICES,
    build_dual_arm_monitor_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--source-hash-index", type=Path, required=True)
    parser.add_argument("--source-done", type=Path, required=True)
    parser.add_argument(
        "--mapping-verdict", choices=DUAL_MAPPING_CHOICES, required=True
    )
    parser.add_argument("--both-leaders-move-freely", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_dual_arm_monitor_receipt(
        source_run_id=args.source_run_id,
        gate_path=args.gate,
        source_hash_index_path=args.source_hash_index,
        source_done_path=args.source_done,
        mapping_verdict=args.mapping_verdict,
        both_leaders_move_freely_after_monitor=(
            args.both_leaders_move_freely
        ),
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("dual-arm monitor receipt was not accepted")


if __name__ == "__main__":
    main()
