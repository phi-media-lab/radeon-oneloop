#!/usr/bin/env python3
"""Validate a Genesis fixed-camera render against the HIL median image."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from .fixed_workspace import detect_target_quads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def match_centers(
    reference: np.ndarray, simulation: np.ndarray
) -> list[tuple[int, int, float]]:
    """Return the minimum-cost one-to-one center matching."""
    reference = np.asarray(reference, dtype=np.float64)
    simulation = np.asarray(simulation, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1:] != (2,):
        raise ValueError("reference centers must have shape (N, 2)")
    if simulation.ndim != 2 or simulation.shape[1:] != (2,):
        raise ValueError("simulation centers must have shape (N, 2)")
    if len(simulation) == 0 or len(simulation) > len(reference):
        raise ValueError("simulation must contain 1..N visible centers")
    best: list[tuple[int, int, float]] | None = None
    best_cost = float("inf")
    for reference_indexes in itertools.permutations(
        range(len(reference)), len(simulation)
    ):
        matches = [
            (
                reference_index,
                simulation_index,
                float(
                    np.linalg.norm(
                        reference[reference_index] - simulation[simulation_index]
                    )
                ),
            )
            for simulation_index, reference_index in enumerate(reference_indexes)
        ]
        cost = sum(match[2] for match in matches)
        if cost < best_cost:
            best, best_cost = matches, cost
    assert best is not None
    return sorted(best)


def validate(args: argparse.Namespace) -> dict[str, object]:
    reference_path = args.reference.resolve()
    simulation_path = args.simulation.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    simulation_image = cv2.imread(str(simulation_path), cv2.IMREAD_COLOR)
    if reference_image is None or simulation_image is None:
        raise ValueError("reference and simulation images must both decode")
    if reference_image.shape != simulation_image.shape:
        raise ValueError("reference and simulation image shapes differ")
    reference_targets = detect_target_quads(reference_image, min_area=args.min_area)
    simulation_targets = detect_target_quads(simulation_image, min_area=args.min_area)
    reference_centers = np.asarray(
        [target["center_px"] for target in reference_targets], dtype=np.float64
    )
    simulation_centers = np.asarray(
        [target["center_px"] for target in simulation_targets], dtype=np.float64
    )
    if len(reference_centers) < args.min_matches:
        raise RuntimeError("too few reference targets")
    if not args.min_matches <= len(simulation_centers) <= len(reference_centers):
        raise RuntimeError("too few or too many simulation targets")
    matches = match_centers(reference_centers, simulation_centers)
    errors = np.asarray([match[2] for match in matches], dtype=np.float64)
    p95 = float(np.percentile(errors, 95))
    accepted = p95 <= args.max_center_p95_px
    report = {
        "schema_version": "radeon_oneloop.hil_front_render_alignment.v1",
        "formal": False,
        "status": "accepted_p0_render_alignment" if accepted else "rejected",
        "accepted": accepted,
        "reference_sha256": sha256_file(reference_path),
        "simulation_sha256": sha256_file(simulation_path),
        "image_size_px": [reference_image.shape[1], reference_image.shape[0]],
        "target_counts": {
            "reference": len(reference_targets),
            "simulation_fully_visible": len(simulation_targets),
        },
        "matches": [
            {
                "reference_index": reference_index,
                "simulation_index": simulation_index,
                "reference_center_px": reference_centers[reference_index].tolist(),
                "simulation_center_px": simulation_centers[simulation_index].tolist(),
                "center_error_px": error,
            }
            for reference_index, simulation_index, error in matches
        ],
        "center_error_px": {
            "mean": float(np.mean(errors)),
            "p95": p95,
            "max": float(np.max(errors)),
            "quality_gate_p95": args.max_center_p95_px,
        },
        "limitations": [
            "the metric validates rendered target registration, not photorealism",
            "targets occluded by simulated robot geometry are excluded by the full-quad detector",
        ],
    }
    metrics = output / "metrics.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "hashes.sha256").write_text(
        f"{sha256_file(metrics)}  {metrics.name}\n", encoding="utf-8"
    )
    (output / ("DONE" if accepted else "FAILED")).touch()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area", type=float, default=200.0)
    parser.add_argument("--min-matches", type=int, default=2)
    parser.add_argument("--max-center-p95-px", type=float, default=10.0)
    args = parser.parse_args()
    report = validate(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
