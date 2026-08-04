#!/usr/bin/env python3
"""Seal the operator judgment for the first single-arm physical stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .haptic_stage_receipt import (
    PERCEPTION_CHOICES,
    build_single_arm_physical_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--source-hash-index", type=Path, required=True)
    parser.add_argument("--source-done", type=Path, required=True)
    parser.add_argument("--perception", choices=PERCEPTION_CHOICES, required=True)
    parser.add_argument("--no-cross-joint-instability", action="store_true")
    parser.add_argument("--leader-moves-freely-after-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_single_arm_physical_receipt(
        source_run_id=args.source_run_id,
        gate_path=args.gate,
        source_hash_index_path=args.source_hash_index,
        source_done_path=args.source_done,
        perception=args.perception,
        no_cross_joint_instability=args.no_cross_joint_instability,
        leader_moves_freely_after_test=args.leader_moves_freely_after_test,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("single-arm physical receipt was not accepted")


if __name__ == "__main__":
    main()
