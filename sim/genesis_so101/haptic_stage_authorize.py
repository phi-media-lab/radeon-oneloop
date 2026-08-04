#!/usr/bin/env python3
"""Verify a sealed haptic-stage receipt and emit a downstream authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .haptic_stage_receipt import authorize_receipt_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--hash-index", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    parser.add_argument("--target-stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authorization = authorize_receipt_bundle(
        receipt_path=args.receipt,
        hash_index_path=args.hash_index,
        done_path=args.done,
        target_stage=args.target_stage,
    )
    payload = json.dumps(authorization, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
