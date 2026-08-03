"""Deterministic checkpoint selection and lineage hashing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def select(metrics_path: Path, output_dir: Path, candidate_steps: list[int]) -> dict[str, Any]:
    """Select best validation success, then lower p95 latency, then earlier step."""
    value = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = value.get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError("metrics must contain a checkpoints list")
    eligible = [row for row in rows if int(row["step"]) in candidate_steps]
    if {int(row["step"]) for row in eligible} != set(candidate_steps):
        raise ValueError("metrics do not contain every predeclared candidate step")
    winner = max(
        eligible,
        key=lambda row: (
            int(row["successes"]),
            -float(row["inference_p95_ms"]),
            -int(row["step"]),
        ),
    )
    checkpoint = output_dir / "checkpoints" / f"{int(winner['step']):06d}" / "pretrained_model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return {
        "schema_version": "radeon_oneloop.checkpoint_selection.v1",
        "rule": "max_validation_success_then_min_p95_then_earliest_step",
        "candidate_steps": candidate_steps,
        "selected_step": int(winner["step"]),
        "selected_checkpoint": str(checkpoint),
        "checkpoint_sha256": tree_hash(checkpoint),
        "selected_metrics": winner,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-steps", default="2000,5000,10000")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = select(
        args.metrics,
        args.output_dir,
        [int(value) for value in args.candidate_steps.split(",")],
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

