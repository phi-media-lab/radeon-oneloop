from __future__ import annotations

import argparse

import pytest

from gaussian.record_seva_orbit_review import (
    ACCEPTED,
    REJECTED,
    REQUIRED_CHECKS,
    REVIEWER_ROLES,
    numeric_gates,
    parse_check,
)


def test_parse_check_accepts_only_named_booleans() -> None:
    assert parse_check(f"{REQUIRED_CHECKS[0]}=true") == (REQUIRED_CHECKS[0], True)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_check("not_a_gate=true")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_check(f"{REQUIRED_CHECKS[0]}=yes")
    assert ACCEPTED != REJECTED
    assert "project_owner_human_review" in REVIEWER_ROLES


def test_numeric_gate_evaluation_is_explicit() -> None:
    metrics = {
        "real_anchor_silhouette_iou": {"mean": 0.7, "min": 0.6},
        "adjacent_foreground_iou": {"p05": 0.8},
        "cyclic_seam": {"rgb_mae_over_adjacent_p95": 1.2},
        "foreground_stability": {
            "area_fraction_cv": 0.05,
            "centroid_x_range_normalized": 0.04,
            "centroid_y_range_normalized": 0.03,
        },
    }
    gates = numeric_gates(metrics)
    assert gates
    assert all(value["passed"] for value in gates.values())
    metrics["cyclic_seam"]["rgb_mae_over_adjacent_p95"] = 3.0
    assert numeric_gates(metrics)["seam_over_adjacent_p95"]["passed"] is False
