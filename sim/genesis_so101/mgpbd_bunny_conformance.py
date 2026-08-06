#!/usr/bin/env python3
"""Headless Radeon numerical conformance gate for the public MGPBD bunny.

This module deliberately imports neither Genesis nor any live/robot module. It
uses pinned upstream TetGen data to exercise the clean-room Torch projector in
isolation, then records the limits of the resulting claim.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

from .mgpbd_reference_io import (
    REFERENCE_COMMIT,
    REFERENCE_MODELS,
    load_reference_mesh,
    signed_six_volumes,
    verify_reference_scene,
)
from .mgpbd_tet import MGPBDTetConfig, VolumetricMGPBDProjector


def squash_y(positions: np.ndarray) -> np.ndarray:
    result = np.asarray(positions, dtype=np.float32).copy()
    result[:, 1] = float(np.min(result[:, 1]))
    return result


def positive_height_squash(
    positions: np.ndarray, height_ratio: float
) -> np.ndarray:
    """Compress Y affinely while retaining a strictly positive tet volume."""

    if not 0.0 < height_ratio < 1.0:
        raise ValueError("positive squash height ratio must be in (0, 1)")
    result = np.asarray(positions, dtype=np.float32).copy()
    minimum = float(np.min(result[:, 1]))
    result[:, 1] = minimum + height_ratio * (result[:, 1] - minimum)
    return result


def rigid_aligned_errors(
    current: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    current64 = np.asarray(current, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    current_centered = current64 - current64.mean(axis=0)
    reference_centered = reference64 - reference64.mean(axis=0)
    covariance = current_centered.T @ reference_centered
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    aligned = current_centered @ rotation + reference64.mean(axis=0)
    errors = np.linalg.norm(aligned - reference64, axis=1)
    diagonal = float(np.linalg.norm(np.ptp(reference64, axis=0)))
    return {
        "rms_sim_units": float(np.sqrt(np.mean(errors * errors))),
        "p95_sim_units": float(np.percentile(errors, 95)),
        "maximum_sim_units": float(np.max(errors)),
        "rms_over_rest_bbox_diagonal": float(
            np.sqrt(np.mean(errors * errors)) / diagonal
        ),
        "p95_over_rest_bbox_diagonal": float(np.percentile(errors, 95) / diagonal),
        "maximum_over_rest_bbox_diagonal": float(np.max(errors) / diagonal),
    }


def boundary_area_metrics(
    positions: np.ndarray, faces: np.ndarray, rest_areas: np.ndarray
) -> dict[str, float | int]:
    triangles = np.asarray(positions, dtype=np.float64)[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    ratio = areas / np.maximum(rest_areas, np.finfo(np.float64).tiny)
    return {
        "minimum_area_ratio": float(np.min(ratio)),
        "p01_area_ratio": float(np.percentile(ratio, 1)),
        "degenerate_faces": int(np.count_nonzero(ratio <= 1.0e-8)),
    }


def public_matrix_ua_profile_and_builder(
    profile: str, frame_history: list[dict[str, object]]
) -> bool:
    """Return true only for the exact public-profile hierarchy contract."""

    return profile == "public_matrix_ua" and bool(frame_history) and all(
        frame["solver"]["amg_hierarchy_builder"]
        == "PyAMG_plain_UA_smooth_none_clean_room"
        for frame in frame_history
    )


def arap_recovery_progress(
    initial_state: dict[str, object], final_state: dict[str, object]
) -> bool:
    """Require both aggregate and worst-element strain to improve."""

    return (
        float(final_state["arap_l2"]) < float(initial_state["arap_l2"])
        and float(final_state["arap_maximum"])
        < float(initial_state["arap_maximum"])
        and float(final_state["extent_ratio_to_rest"][1]) > 0.0
    )


def multiplier_contract_consistent(
    record: dict[str, object], *, rejected_policy: str
) -> bool:
    """Verify the recorded multiplier transaction, not just the position step."""

    if rejected_policy not in {"retain_full", "rollback"}:
        raise ValueError("invalid rejected multiplier contract")
    rejected = bool(record.get("line_search_rejected"))
    expected_policy = (
        "retain_full_public_multiplier"
        if rejected and rejected_policy == "retain_full"
        else (
            "rollback_multiplier_with_position"
            if rejected
            else "accepted_full_trial_multiplier"
        )
    )
    expected_fraction = (
        0.0 if rejected and rejected_policy == "rollback" else 1.0
    )
    return bool(
        record.get("lagrangian_fraction_matches_observed", False)
        and float(
            record.get("lagrangian_transaction_relative_error", np.inf)
        )
        <= 1.0e-6
        and record.get("lagrangian_acceptance_policy") == expected_policy
        and np.isclose(
            float(record.get("lagrangian_update_fraction", np.nan)),
            expected_fraction,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    )


def public_line_search_record_consistent(record: dict[str, object]) -> bool:
    """Validate both position merit and public full-multiplier semantics."""

    rejected = bool(record.get("line_search_rejected"))
    position_consistent = (
        rejected
        and float(record.get("line_search_step", np.nan)) == 0.0
        and np.isclose(
            float(record.get("line_search_objective_after", np.nan)),
            float(record.get("line_search_objective_before", np.nan)),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    ) or (
        not rejected
        and float(record.get("line_search_step", np.nan)) > 0.0
        and float(record.get("line_search_objective_after", np.nan))
        < float(record.get("line_search_objective_before", np.nan))
    )
    return bool(
        position_consistent
        and multiplier_contract_consistent(
            record, rejected_policy="retain_full"
        )
    )


def _finite_record_number(
    record: dict[str, object], key: str
) -> float | None:
    """Read one finite scalar without allowing booleans to masquerade as it."""

    value = record.get(key)
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _nonnegative_record_integer(
    record: dict[str, object], key: str, *, minimum: int = 0
) -> bool:
    value = _finite_record_number(record, key)
    return bool(
        value is not None
        and value >= minimum
        and value == np.floor(value)
    )


def _sqp_receipt(record: dict[str, object]) -> dict[str, object] | None:
    receipt = record.get("sqp_direction")
    return receipt if isinstance(receipt, dict) else None


def _soc_admm_receipt(record: dict[str, object]) -> dict[str, object] | None:
    """Return the integrated SOC receipt only when it is structurally present."""

    receipt = record.get("soc_admm_direction")
    return receipt if isinstance(receipt, dict) else None


def soc_admm_transaction_consistent(record: dict[str, object]) -> bool:
    """Verify the SOC position/multiplier transaction and Armijo acceptance.

    The integrated SOC direction is a complete nonlinear-outer direction, not
    the isolated kernel smoke.  The outer line search may shorten that
    direction, but the exact same scalar must update both positions and the
    reconstructed MGPBD multiplier.  A rejected trial must roll both back.
    """

    rejected_value = record.get("line_search_rejected")
    receipt = _soc_admm_receipt(record)
    if not isinstance(rejected_value, (bool, np.bool_)) or receipt is None:
        return False
    rejected = bool(rejected_value)
    scalars = {
        name: _finite_record_number(source, name)
        for source, name in (
            (record, "line_search_step"),
            (record, "lagrangian_update_fraction"),
            (record, "lagrangian_transaction_relative_error"),
            (record, "correction_global_scale"),
            (receipt, "accepted_step"),
            (receipt, "accepted_multiplier_fraction"),
            (receipt, "armijo_coefficient"),
            (receipt, "armijo_merit_before"),
            (receipt, "armijo_merit_after"),
            (receipt, "merit_slope"),
            (receipt, "armijo_rhs"),
        )
    }
    if any(value is None for value in scalars.values()):
        return False
    step = float(scalars["line_search_step"])
    multiplier_fraction = float(scalars["lagrangian_update_fraction"])
    receipt_step = float(scalars["accepted_step"])
    receipt_fraction = float(scalars["accepted_multiplier_fraction"])
    coefficient = float(scalars["armijo_coefficient"])
    merit_before = float(scalars["armijo_merit_before"])
    merit_after = float(scalars["armijo_merit_after"])
    merit_slope = float(scalars["merit_slope"])
    armijo_rhs = float(scalars["armijo_rhs"])
    expected_step = 0.0 if rejected else step
    expected_policy = (
        "rollback_multiplier_with_position"
        if rejected
        else "accepted_scaled_trial_multiplier"
    )
    expected_rhs = merit_before + coefficient * expected_step * merit_slope
    armijo_consistent = bool(
        np.isclose(armijo_rhs, expected_rhs, rtol=1.0e-6, atol=1.0e-7)
        and (
            (
                rejected
                and receipt.get("armijo_satisfied") is False
                and np.isclose(
                    merit_after, merit_before, rtol=1.0e-6, atol=1.0e-7
                )
            )
            or (
                not rejected
                and receipt.get("armijo_satisfied") is True
                and merit_after <= armijo_rhs
            )
        )
    )
    return bool(
        record.get("direction_backend") == "soc_admm"
        and record.get("legacy_direction_pcg_skipped") is True
        and float(scalars["correction_global_scale"]) == 1.0
        and 0.0 <= float(scalars["lagrangian_transaction_relative_error"])
        <= 1.0e-6
        and record.get("lagrangian_fraction_matches_observed") is True
        and record.get("position_and_lagrangian_step_accepted_atomically")
        is True
        and receipt.get("coupled_position_multiplier_transaction") is True
        and receipt.get("rolled_back_atomically") is True
        and record.get("lagrangian_acceptance_policy") == expected_policy
        and receipt.get("multiplier_acceptance_policy") == expected_policy
        and (step == 0.0 if rejected else 0.0 < step <= 1.0)
        and np.isclose(step, expected_step, rtol=1.0e-6, atol=1.0e-7)
        and np.isclose(
            multiplier_fraction,
            expected_step,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        and np.isclose(
            receipt_step, expected_step, rtol=1.0e-6, atol=1.0e-7
        )
        and np.isclose(
            receipt_fraction,
            expected_step,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        and 0.0 < coefficient < 0.5
        and merit_before >= 0.0
        and merit_after >= 0.0
        and (
            merit_slope < 0.0
            or (
                merit_slope == 0.0
                and merit_before == 0.0
                and merit_after == 0.0
            )
        )
        and armijo_consistent
    )


def soc_admm_numerical_receipt_consistent(
    record: dict[str, object], config: MGPBDTetConfig
) -> bool:
    """Fail closed on an incomplete integrated SOC-ADMM direction receipt."""

    receipt = _soc_admm_receipt(record)
    if receipt is None:
        return False
    nested = receipt.get("configuration")
    checks = receipt.get("checks")
    pcg_receipts = receipt.get("pcg_receipts")
    normal = receipt.get("normal_cone")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (nested, dict),
            (checks, dict),
            (pcg_receipts, list),
            (normal, dict),
        )
    ):
        return False
    assert isinstance(nested, dict)
    assert isinstance(checks, dict)
    assert isinstance(pcg_receipts, list)
    assert isinstance(normal, dict)
    if normal.get("dual_convention") != (
        "physical_y_equals_beta_times_scaled_dual"
    ):
        return False
    expected_checks = {
        "admm_converged",
        "pcg_true_residuals_satisfied",
        "admm_primal_satisfied",
        "admm_dual_satisfied",
        "stationarity_satisfied",
        "normal_cone_satisfied",
        "soc_projection_feasible",
        "determinant_proof_satisfied",
        "coupled_material_satisfied",
        "true_arap_satisfied",
        "true_determinant_satisfied",
        "finite_candidate",
        "objective_not_above_zero_direction",
    }
    if not expected_checks.issubset(checks) or not all(
        checks.get(name) is True for name in expected_checks
    ):
        return False

    # A warm FP32 -> FP64 continuation is allowed only as an internal
    # numerical refinement.  The accepted direction still enters the FP32
    # projector, so its receipt is incomplete unless the solver cast it back
    # exactly once and repeated every safety/material gate in that dtype.
    # Keep this conditional: native-FP64 and FP32 solves that never encounter
    # a precision floor legitimately have no accepted-dtype re-audit block.
    precision_continuation_active = receipt.get("precision_continuation_active")
    precision_continuation_events = receipt.get("precision_continuation_events")
    precision_continuation_started = receipt.get(
        "precision_continuation_started_iteration"
    )
    if not isinstance(
        precision_continuation_active, (bool, np.bool_)
    ) or not isinstance(precision_continuation_events, list):
        return False
    if bool(precision_continuation_active):
        accepted_dtype_reaudit = receipt.get("accepted_dtype_reaudit")
        if (
            not _nonnegative_record_integer(
                receipt, "precision_continuation_started_iteration", minimum=1
            )
            or not precision_continuation_events
            or not isinstance(accepted_dtype_reaudit, dict)
            or accepted_dtype_reaudit.get("passed") is not True
            or checks.get("accepted_dtype_reaudit_satisfied") is not True
        ):
            return False
        accepted_checks = accepted_dtype_reaudit.get("checks")
        if not isinstance(accepted_checks, dict) or not accepted_checks or not all(
            value is True for value in accepted_checks.values()
        ):
            return False
    elif precision_continuation_started is not None or precision_continuation_events:
        return False

    numbers = {
        name: _finite_record_number(receipt, name)
        for name in (
            "admm_primal_residual_maximum",
            "admm_dual_residual_relative",
            "stationarity_relative",
            "safety_proof_radius_maximum",
            "true_arap_maximum",
            "minimum_signed_volume_ratio",
            "coupled_material_residual_relative",
            "soc_z_violation_maximum",
            "initial_objective",
            "final_objective",
            "direction_l2",
            "delta_lambda_l2",
        )
    }
    normal_residual = _finite_record_number(normal, "gate_residual")
    if any(value is None for value in numbers.values()) or normal_residual is None:
        return False
    count_fields = (
        ("admm_iterations", 1),
        ("pcg_solves", 1),
        ("pcg_iterations_total", 0),
        ("consecutive_gate_passes_final", 1),
        ("adaptive_beta_update_count", 0),
        ("inverted_or_collapsed_tetrahedra", 0),
    )
    if not all(
        _nonnegative_record_integer(receipt, name, minimum=minimum)
        for name, minimum in count_fields
    ):
        return False
    if int(receipt["pcg_solves"]) != len(pcg_receipts):
        return False
    if any(
        not isinstance(item, dict)
        or item.get("converged") is not True
        or (
            (ratio := _finite_record_number(item, "true_residual_to_target"))
            is None
        )
        or ratio > 1.0 + 1.0e-6
        for item in pcg_receipts
    ):
        return False

    config_pairs = {
        "work_radius": config.soc_admm_work_radius,
        "true_arap_maximum": config.soc_admm_true_arap_maximum,
        "minimum_signed_volume_ratio": (
            config.soc_admm_minimum_signed_volume_ratio
        ),
        "admm_primal_tolerance": config.soc_admm_primal_tolerance,
        "admm_dual_relative_tolerance": (
            config.soc_admm_dual_relative_tolerance
        ),
        "stationarity_relative_tolerance": (
            config.soc_admm_stationarity_relative_tolerance
        ),
        "normal_cone_tolerance": config.soc_admm_normal_cone_tolerance,
        "coupled_material_relative_tolerance": (
            config.soc_admm_coupled_material_relative_tolerance
        ),
        "pcg_relative_tolerance": config.soc_admm_pcg_relative_tolerance,
    }
    for name, expected in config_pairs.items():
        observed = _finite_record_number(nested, name)
        if observed is None or not np.isclose(
            observed, expected, rtol=1.0e-12, atol=0.0
        ):
            return False
    proof_radius = _finite_record_number(nested, "proof_radius")
    if proof_radius is None:
        return False
    return bool(
        receipt.get("backend") == "Torch_ROCm_matrix_free_SOC_ADMM"
        and receipt.get("converged") is True
        and receipt.get("fallback_used") is False
        and receipt.get("passed") is True
        and receipt.get("failure") is None
        and int(receipt["admm_iterations"])
        <= config.soc_admm_maximum_iterations
        and int(receipt["consecutive_gate_passes_final"])
        >= config.soc_admm_required_consecutive_gate_passes
        and float(numbers["admm_primal_residual_maximum"])
        <= config.soc_admm_primal_tolerance
        and float(numbers["admm_dual_residual_relative"])
        <= config.soc_admm_dual_relative_tolerance
        and float(numbers["stationarity_relative"])
        <= config.soc_admm_stationarity_relative_tolerance
        and normal_residual <= config.soc_admm_normal_cone_tolerance
        and float(numbers["safety_proof_radius_maximum"]) <= proof_radius
        and float(numbers["true_arap_maximum"])
        <= config.soc_admm_true_arap_maximum
        and float(numbers["minimum_signed_volume_ratio"])
        >= config.soc_admm_minimum_signed_volume_ratio
        and int(receipt["inverted_or_collapsed_tetrahedra"]) == 0
        and float(numbers["coupled_material_residual_relative"])
        <= config.soc_admm_coupled_material_relative_tolerance
        and float(numbers["soc_z_violation_maximum"])
        <= config.soc_admm_primal_tolerance
        and float(numbers["initial_objective"]) >= 0.0
        and float(numbers["final_objective"])
        <= float(numbers["initial_objective"])
        + max(
            1.0e-12,
            config.soc_admm_stationarity_relative_tolerance
            * max(float(numbers["initial_objective"]), 1.0),
        )
        and float(numbers["direction_l2"]) >= 0.0
        and float(numbers["delta_lambda_l2"]) >= 0.0
    )


def soc_admm_outer_receipt_consistent(
    record: dict[str, object], config: MGPBDTetConfig
) -> bool:
    """Require both the SOC numerical proof and its accepted outer transaction."""

    return bool(
        soc_admm_numerical_receipt_consistent(record, config)
        and soc_admm_transaction_consistent(record)
    )


def scaled_sqp_multiplier_contract_consistent(
    record: dict[str, object],
) -> bool:
    """Verify one scaled position/multiplier SQP transaction end to end."""

    rejected_value = record.get("line_search_rejected")
    receipt = _sqp_receipt(record)
    if not isinstance(rejected_value, (bool, np.bool_)) or receipt is None:
        return False
    rejected = bool(rejected_value)
    step = _finite_record_number(record, "line_search_step")
    fraction = _finite_record_number(record, "lagrangian_update_fraction")
    global_scale = _finite_record_number(record, "correction_global_scale")
    receipt_step = _finite_record_number(receipt, "accepted_step")
    receipt_fraction = _finite_record_number(
        receipt, "accepted_multiplier_fraction"
    )
    transaction_error = _finite_record_number(
        record, "lagrangian_transaction_relative_error"
    )
    if None in {
        step,
        fraction,
        global_scale,
        receipt_step,
        receipt_fraction,
        transaction_error,
    }:
        return False
    expected_policy = (
        "rollback_multiplier_with_position"
        if rejected
        else "accepted_scaled_trial_multiplier"
    )
    expected_step = 0.0 if rejected else step
    expected_fraction = expected_step
    return bool(
        global_scale == 1.0
        and 0.0 <= transaction_error <= 1.0e-6
        and record.get("lagrangian_fraction_matches_observed") is True
        and record.get("position_and_lagrangian_step_accepted_atomically")
        is True
        and record.get("lagrangian_acceptance_policy") == expected_policy
        and receipt.get("multiplier_acceptance_policy") == expected_policy
        and receipt.get("coupled_position_multiplier_transaction") is True
        and receipt.get("rolled_back_atomically") is True
        and (step == 0.0 if rejected else 0.0 < step <= 1.0)
        and np.isclose(
            fraction, expected_fraction, rtol=1.0e-6, atol=1.0e-7
        )
        and np.isclose(
            receipt_step, expected_step, rtol=1.0e-6, atol=1.0e-7
        )
        and np.isclose(
            receipt_fraction,
            expected_fraction,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    )


def sqp_armijo_and_atomic_receipt_consistent(
    record: dict[str, object], config: MGPBDTetConfig
) -> bool:
    """Accept an Armijo step or a complete position/multiplier rollback."""

    receipt = _sqp_receipt(record)
    if receipt is None or not scaled_sqp_multiplier_contract_consistent(record):
        return False
    coefficient = _finite_record_number(receipt, "armijo_coefficient")
    merit_before = _finite_record_number(receipt, "armijo_merit_before")
    merit_after = _finite_record_number(receipt, "armijo_merit_after")
    merit_slope = _finite_record_number(receipt, "merit_slope")
    recorded_rhs = _finite_record_number(receipt, "armijo_rhs")
    step = _finite_record_number(receipt, "accepted_step")
    if None in {
        coefficient,
        merit_before,
        merit_after,
        merit_slope,
        recorded_rhs,
        step,
    }:
        return False
    if (
        coefficient != config.sqp_armijo_coefficient
        or merit_before < 0.0
        or merit_after < 0.0
        or merit_slope >= 0.0
    ):
        return False
    expected_rhs = merit_before + coefficient * step * merit_slope
    if not np.isclose(
        recorded_rhs, expected_rhs, rtol=1.0e-6, atol=1.0e-7
    ):
        return False
    rejected = bool(record["line_search_rejected"])
    if rejected:
        return bool(
            receipt.get("armijo_satisfied") is False
            and np.isclose(
                merit_after, merit_before, rtol=1.0e-6, atol=1.0e-7
            )
        )
    return bool(
        receipt.get("armijo_satisfied") is True
        and merit_after <= recorded_rhs
    )


def _sqp_receipt_tolerances(
    receipt: dict[str, object], config: MGPBDTetConfig
) -> tuple[float, float, float, float] | None:
    nested = receipt.get("configuration")
    nested = nested if isinstance(nested, dict) else {}

    def configured(names: tuple[str, ...], fallback: float) -> float | None:
        for name in names:
            if name in nested:
                return _finite_record_number(nested, name)
        return float(fallback)

    primal = configured(
        ("primal_tolerance", "sqp_primal_tolerance"),
        config.sqp_primal_tolerance,
    )
    dual = configured(
        ("dual_tolerance", "sqp_dual_tolerance"),
        config.sqp_dual_tolerance,
    )
    kkt = configured(
        ("kkt_relative_tolerance",), config.relative_residual
    )
    auxiliary = configured(
        ("auxiliary_relative_residual_tolerance", "relative_residual"),
        config.relative_residual,
    )
    if (
        primal is None
        or dual is None
        or kkt is None
        or auxiliary is None
        or primal <= 0.0
        or dual < 0.0
        or kkt <= 0.0
        or auxiliary <= 0.0
    ):
        return None
    return primal, dual, kkt, auxiliary


def sqp_numerical_receipt_consistent(
    record: dict[str, object], config: MGPBDTetConfig
) -> bool:
    """Fail closed on incomplete SQP/KKT, safety-cut, or residual evidence."""

    receipt = _sqp_receipt(record)
    if receipt is None:
        return False
    tolerances = _sqp_receipt_tolerances(receipt, config)
    if tolerances is None:
        return False
    primal_tolerance, dual_tolerance, kkt_tolerance, auxiliary_tolerance = (
        tolerances
    )
    required_numbers = {
        key: _finite_record_number(receipt, key)
        for key in (
            "final_maximum_linearized_violation",
            "minimum_multiplier",
            "maximum_auxiliary_true_relative_residual",
            "active_equality_residual_maximum",
            "complementarity_maximum",
            "stationarity_relative",
            "schur_residual_relative",
            "schur_minimum_eigenvalue",
            "coupled_linearized_residual_relative",
            "linearized_material_residual_l2",
            "direction_change_l2",
        )
    }
    if any(value is None for value in required_numbers.values()):
        return False
    count_keys = (
        "active_constraints",
        "auxiliary_linear_solves",
        "auxiliary_columns_computed",
        "auxiliary_initial_linear_solves",
        "auxiliary_refinement_linear_solves",
        "auxiliary_pcg_iterations_total",
        "auxiliary_zero_rhs_columns",
        "auxiliary_final_active_columns",
    )
    required_counts = {
        key: _finite_record_number(receipt, key) for key in count_keys
    }
    if not all(
        _nonnegative_record_integer(receipt, key, minimum=minimum)
        for key, minimum in (
            ("active_set_iterations", 1),
            ("active_constraints", 0),
            ("auxiliary_linear_solves", 0),
            ("auxiliary_columns_computed", 0),
            ("auxiliary_initial_linear_solves", 0),
            ("auxiliary_refinement_linear_solves", 0),
            ("auxiliary_pcg_iterations_total", 0),
            ("auxiliary_zero_rhs_columns", 0),
            ("auxiliary_final_active_columns", 0),
            ("nonlinear_safety_resolves", 1),
            ("nonlinear_strain_cuts", 0),
            ("nonlinear_determinant_cuts", 0),
        )
    ):
        return False
    return bool(
        receipt.get("enabled") is True
        and receipt.get("backend")
        == "Torch_ROCm_MGPCG_Schur_active_set"
        and receipt.get("converged") is True
        and receipt.get("fallback_used") is False
        and receipt.get("kkt_passed") is True
        and receipt.get("full_direction_safety_feasible") is True
        and required_counts["auxiliary_linear_solves"]
        == required_counts["auxiliary_initial_linear_solves"]
        + required_counts["auxiliary_refinement_linear_solves"]
        and required_counts["auxiliary_columns_computed"]
        == required_counts["auxiliary_initial_linear_solves"]
        + required_counts["auxiliary_zero_rhs_columns"]
        and required_counts["auxiliary_final_active_columns"]
        == required_counts["active_constraints"]
        and required_counts["auxiliary_final_active_columns"]
        <= required_counts["auxiliary_columns_computed"]
        and required_numbers["final_maximum_linearized_violation"]
        <= primal_tolerance
        and required_numbers["minimum_multiplier"] >= -dual_tolerance
        and 0.0
        <= required_numbers["maximum_auxiliary_true_relative_residual"]
        <= auxiliary_tolerance
        and 0.0
        <= required_numbers["active_equality_residual_maximum"]
        <= kkt_tolerance
        and 0.0
        <= required_numbers["complementarity_maximum"]
        <= kkt_tolerance
        and 0.0
        <= required_numbers["stationarity_relative"]
        <= kkt_tolerance
        and 0.0
        <= required_numbers["schur_residual_relative"]
        <= kkt_tolerance
        and required_numbers["schur_minimum_eigenvalue"] >= -kkt_tolerance
        and 0.0
        <= required_numbers["coupled_linearized_residual_relative"]
        <= kkt_tolerance
        and required_numbers["linearized_material_residual_l2"] >= 0.0
        and required_numbers["direction_change_l2"] >= 0.0
    )


def sqp_receipt_contract_consistent(
    record: dict[str, object], config: MGPBDTetConfig
) -> bool:
    """Require a complete transaction, globalization, and numerical receipt."""

    return bool(
        scaled_sqp_multiplier_contract_consistent(record)
        and sqp_armijo_and_atomic_receipt_consistent(record, config)
        and sqp_numerical_receipt_consistent(record, config)
    )


def write_boundary_obj(path: Path, positions: np.ndarray, faces: np.ndarray) -> None:
    lines = [
        *(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in positions),
        *(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces),
    ]
    path.write_text("".join(lines), encoding="utf-8")


def synchronize(device: object) -> None:
    import torch

    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def frame_state_metrics(
    positions: np.ndarray,
    velocity: np.ndarray,
    rest_positions: np.ndarray,
    elements: np.ndarray,
    rest_signed_six: np.ndarray,
    boundary: np.ndarray,
    rest_boundary_areas: np.ndarray,
    constraints: np.ndarray,
) -> dict[str, object]:
    signed_ratio = signed_six_volumes(positions, elements) / rest_signed_six
    extents = np.ptp(positions.astype(np.float64), axis=0)
    rest_extents = np.ptp(rest_positions.astype(np.float64), axis=0)
    area = boundary_area_metrics(positions, boundary, rest_boundary_areas)
    return {
        "finite_positions": bool(np.isfinite(positions).all()),
        "finite_velocity": bool(np.isfinite(velocity).all()),
        "center_sim_units": positions.astype(np.float64).mean(axis=0).tolist(),
        "extents_sim_units": extents.tolist(),
        "extent_ratio_to_rest": (extents / rest_extents).tolist(),
        "arap_l2": float(np.linalg.norm(constraints.astype(np.float64))),
        "arap_maximum": float(np.max(constraints)),
        "arap_p95": float(np.percentile(constraints, 95)),
        "minimum_signed_volume_ratio": float(np.min(signed_ratio)),
        "p01_signed_volume_ratio": float(np.percentile(signed_ratio, 1)),
        "inverted_tetrahedra": int(np.count_nonzero(signed_ratio < 0.0)),
        "collapsed_tetrahedra": int(np.count_nonzero(signed_ratio <= 1.0e-8)),
        "boundary": area,
    }


def direct_first_direction_oracle(
    projector: VolumetricMGPBDProjector,
    positions: object,
    elements: np.ndarray,
    rest_signed_six: np.ndarray,
) -> dict[str, object]:
    """Solve the first dual system directly on CPU and audit orientation."""

    from scipy.sparse.linalg import spsolve

    started = time.perf_counter()
    constraints, gradients, _active = projector.constraints_and_gradients(positions)
    matrix = projector._assemble_cpu_matrix(gradients)  # audited oracle only
    rhs = -constraints.detach().cpu().numpy().astype(np.float64)
    delta_lambda = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
    true_relative = float(
        np.linalg.norm(rhs - matrix @ delta_lambda)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
    )
    gradients_np = gradients.detach().cpu().numpy().astype(np.float64)
    correction = np.zeros((len(projector.rest_positions), 3), dtype=np.float64)
    for local_index in range(4):
        vertices = elements[:, local_index]
        contribution = (
            projector.inverse_mass_np[vertices, None]
            * delta_lambda[:, None]
            * gradients_np[:, local_index]
        )
        np.add.at(correction, vertices, contribution)
    positions_np = positions.detach().cpu().numpy().astype(np.float64)
    current_ratio = (
        signed_six_volumes(positions_np, elements) / rest_signed_six
    )
    fractions = (1.0e-9, 1.0e-6, 1.0e-3, 0.1, 0.25, 0.5, 1.0)
    trials: list[dict[str, float | int]] = []
    for fraction in fractions:
        trial_ratio = (
            signed_six_volumes(
                positions_np + fraction * correction, elements
            )
            / rest_signed_six
        )
        trials.append(
            {
                "step": fraction,
                "minimum_signed_volume_ratio": float(np.min(trial_ratio)),
                "inverted_tetrahedra": int(np.count_nonzero(trial_ratio < 0.0)),
                "collapsed_tetrahedra": int(
                    np.count_nonzero(trial_ratio <= 1.0e-8)
                ),
            }
        )
    tiny_ratio = (
        signed_six_volumes(positions_np + 1.0e-9 * correction, elements)
        / rest_signed_six
    )
    directional_derivative = (tiny_ratio - current_ratio) / 1.0e-9
    return {
        "kind": "CPU_SciPy_sparse_direct_first_direction_oracle",
        "Radeon_execution_claim": False,
        "relative_true_residual": true_relative,
        "correction_center_norm": float(
            np.linalg.norm(np.mean(correction, axis=0))
        ),
        "wrong_way_directional_tetrahedra": int(
            np.count_nonzero(directional_derivative < 0.0)
        ),
        "wrong_way_indices_first_32": np.flatnonzero(
            directional_derivative < 0.0
        )[:32].tolist(),
        "trials": trials,
        "elapsed_s": time.perf_counter() - started,
    }


def projector_configuration(profile: str) -> MGPBDTetConfig:
    common = dict(
        dt_s=0.01,
        shear_modulus_pa=1.0e9,
        nonlinear_iterations=20,
        pcg_iterations=100,
        relative_residual=1.0e-5,
        outer_absolute_residual=1.0e-4,
        outer_relative_residual=1.0e-2,
        amg_coarsest_size=400,
        amg_max_levels=10,
        amg_setup_interval_frames=10_000,
        smoother_iterations=2,
        damping_retention=1.0,
        maximum_correction_m=None,
        line_search_acceptance_epsilon=0.0,
        line_search_minimum_step=1.0e-9,
        orientation_diagnostics_enabled=True,
    )
    if profile == "public_matrix_ua":
        return MGPBDTetConfig(
            **common,
            relaxation=1.0,
            line_search_enabled=True,
            line_search_objective="dual",
            amg_hierarchy_mode="matrix_ua",
            symmetric_diagonal_equilibration=False,
            smoother_weight_mode="fine_spectral_radius",
            line_search_rejected_lagrangian_policy="retain_full",
        )
    if profile == "paper_fixed_omega":
        return MGPBDTetConfig(
            **common,
            relaxation=0.1,
            line_search_enabled=False,
            line_search_objective="dual",
            amg_hierarchy_mode="matrix_ua",
            symmetric_diagonal_equilibration=False,
            smoother_weight_mode="fine_spectral_radius",
        )
    if profile == "radeon_equilibrated_matrix_ua":
        return MGPBDTetConfig(
            **common,
            relaxation=1.0,
            line_search_enabled=True,
            line_search_objective="dual",
            amg_hierarchy_mode="matrix_ua",
            symmetric_diagonal_equilibration=True,
            smoother_weight_mode="fine_spectral_radius",
        )
    if profile == "orientation_safe_matrix_ua":
        safe = {
            **common,
            "nonlinear_iterations": 60,
            "pcg_iterations": 1_000,
        }
        return MGPBDTetConfig(
            **safe,
            relaxation=1.0,
            line_search_enabled=True,
            line_search_objective="dual",
            line_search_scale_lagrangian=False,
            line_search_rejected_lagrangian_policy="rollback",
            orientation_guard_enabled=True,
            orientation_guard_minimum_ratio=1.0e-6,
            strain_trust_filter_enabled=True,
            strain_trust_filter_maximum=1.0,
            amg_hierarchy_mode="matrix_ua",
            symmetric_diagonal_equilibration=False,
            smoother_weight_mode="fine_spectral_radius",
        )
    if profile == "orientation_safe_sqp_matrix_ua":
        safe_sqp = {
            **common,
            "nonlinear_iterations": 60,
            "pcg_iterations": 1_000,
        }
        return MGPBDTetConfig(
            **safe_sqp,
            relaxation=1.0,
            line_search_enabled=True,
            line_search_objective="dual",
            line_search_scale_lagrangian=True,
            line_search_rejected_lagrangian_policy="rollback",
            orientation_guard_enabled=True,
            orientation_guard_minimum_ratio=1.0e-6,
            strain_trust_filter_enabled=True,
            strain_trust_filter_maximum=1.0,
            sqp_direction_enabled=True,
            amg_hierarchy_mode="matrix_ua",
            symmetric_diagonal_equilibration=False,
            smoother_weight_mode="fine_spectral_radius",
        )
    if profile == "orientation_safe_soc_matrix_free":
        # These are the first Radeon settings that passed the independent
        # bunny_small direction audit.  P0a2 still runs the complete
        # nonlinear projector loop; this profile must not be confused with
        # the isolated direction smoke.
        safe_soc = {
            **common,
            "nonlinear_iterations": 60,
            # The legacy dual PCG is bypassed.  Keep this valid only because
            # MGPBDTetConfig has a shared positive-count invariant.
            "pcg_iterations": 1,
        }
        return MGPBDTetConfig(
            **safe_soc,
            relaxation=1.0,
            line_search_enabled=True,
            line_search_objective="dual",
            line_search_scale_lagrangian=True,
            line_search_rejected_lagrangian_policy="rollback",
            orientation_guard_enabled=True,
            orientation_guard_minimum_ratio=1.0e-6,
            strain_trust_filter_enabled=True,
            # Preserve recursive feasibility: every accepted outer state must
            # be a valid zero-direction point for the next SOC subproblem.
            strain_trust_filter_maximum=0.989,
            sqp_direction_enabled=False,
            soc_admm_direction_enabled=True,
            soc_admm_beta=1.0e-4,
            # Gate-score residual balancing is intentionally bounded by the
            # largest penalty observed to retain a converged Radeon PCG
            # d-step in the audited bunny_small continuation.  Larger cold or
            # adaptive penalties made Jacobi-PCG fail closed or degraded KKT
            # stationarity; this is a numerical profile bound, not a relaxed
            # feasibility threshold.
            soc_admm_beta_maximum=1.28e-2,
            # Retain the feasibility-producing penalty during KKT polish.
            # A reproducible FP32 residual floor now triggers a warm, full-
            # state FP64 continuation instead of perturbing the ADMM penalty.
            soc_admm_kkt_polish_beta_maximum=None,
            soc_admm_maximum_iterations=2_000,
            soc_admm_required_consecutive_gate_passes=2,
            soc_admm_pcg_maximum_iterations=2_000,
            soc_admm_pcg_relative_tolerance=1.5e-5,
            amg_hierarchy_mode="matrix_ua",
            symmetric_diagonal_equilibration=False,
            smoother_weight_mode="fine_spectral_radius",
        )
    if profile == "topology_ua_ablation":
        return MGPBDTetConfig(
            **common,
            relaxation=1.0,
            line_search_enabled=True,
            line_search_objective="dual",
            amg_hierarchy_mode="topology_ua",
            symmetric_diagonal_equilibration=True,
            smoother_weight_mode="fixed",
        )
    raise ValueError(f"unsupported conformance profile: {profile}")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(REFERENCE_MODELS), required=True)
    parser.add_argument("--mode", choices=("projection", "trajectory"), required=True)
    parser.add_argument(
        "--contract",
        choices=(
            "official_fidelity",
            "orientation_safe_recovery",
            "orientation_safe_sqp_recovery",
            "orientation_safe_soc_recovery",
            "numerical_ablation",
        ),
        default="official_fidelity",
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--profile",
        choices=(
            "public_matrix_ua",
            "paper_fixed_omega",
            "radeon_equilibrated_matrix_ua",
            "orientation_safe_matrix_ua",
            "orientation_safe_sqp_matrix_ua",
            "orientation_safe_soc_matrix_free",
            "topology_ua_ablation",
        ),
        default="public_matrix_ua",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--numerical-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=20_260_806)
    parser.add_argument("--initial-height-ratio", type=float, default=0.25)
    parser.add_argument("--direct-linear-oracle", action="store_true")
    parser.add_argument("--soc-admm-beta", type=float)
    parser.add_argument("--soc-admm-maximum-iterations", type=int)
    parser.add_argument("--soc-admm-pcg-maximum-iterations", type=int)
    parser.add_argument("--soc-admm-pcg-relative-tolerance", type=float)
    parser.add_argument(
        "--soc-admm-required-consecutive-gate-passes", type=int
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "projection" and args.frames != 1:
        raise ValueError("projection mode requires --frames 1")
    if not 1 <= args.frames <= 100:
        raise ValueError("conformance frames must be in [1, 100]")
    if (
        args.contract == "orientation_safe_recovery"
        and args.profile != "orientation_safe_matrix_ua"
    ):
        raise ValueError(
            "orientation-safe contract requires orientation_safe_matrix_ua"
        )
    if (
        args.contract == "orientation_safe_sqp_recovery"
        and args.profile != "orientation_safe_sqp_matrix_ua"
    ):
        raise ValueError(
            "orientation-safe SQP contract requires "
            "orientation_safe_sqp_matrix_ua"
        )
    if (
        args.contract == "orientation_safe_soc_recovery"
        and args.profile != "orientation_safe_soc_matrix_free"
    ):
        raise ValueError(
            "orientation-safe SOC contract requires "
            "orientation_safe_soc_matrix_free"
        )
    if (
        args.contract == "orientation_safe_sqp_recovery"
        and not np.isclose(
            args.initial_height_ratio, 0.25, rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError(
            "orientation-safe SQP recovery requires the audited 0.25 "
            "positive-height squash"
        )
    if (
        args.contract == "orientation_safe_soc_recovery"
        and not np.isclose(
            args.initial_height_ratio, 0.25, rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError(
            "orientation-safe SOC recovery requires the audited 0.25 "
            "positive-height squash"
        )
    if (
        args.contract == "official_fidelity"
        and args.profile != "public_matrix_ua"
    ):
        raise ValueError(
            "official fidelity requires the public_matrix_ua profile"
        )
    if args.contract == "numerical_ablation" and args.profile in {
        "public_matrix_ua",
        "orientation_safe_matrix_ua",
        "orientation_safe_sqp_matrix_ua",
        "orientation_safe_soc_matrix_free",
    }:
        raise ValueError("numerical ablation requires an ablation profile")
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mesh = load_reference_mesh(args.reference_root, args.model)
    scene = verify_reference_scene(args.reference_root)
    scene_payload = json.loads(Path(scene["path"]).read_text(encoding="utf-8"))
    expected_scene = {
        "delta_t": 0.01,
        "mu": 1.0e9,
        "use_gravity": 0,
        "reinit": "squash",
        "maxiter": 20,
        "solver_type": "AMG",
    }
    if any(scene_payload.get(key) != value for key, value in expected_scene.items()):
        raise ValueError("pinned upstream scene no longer matches audited settings")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested Radeon/Torch CUDA-compatible device is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rest_np = mesh.positions
    squashed_np = (
        squash_y(rest_np)
        if args.contract in {"official_fidelity", "numerical_ablation"}
        else positive_height_squash(rest_np, args.initial_height_ratio)
    )
    numerical_dtype = (
        torch.float64 if args.numerical_dtype == "float64" else torch.float32
    )
    rest = torch.as_tensor(rest_np, dtype=numerical_dtype, device=device)
    positions = torch.as_tensor(
        squashed_np, dtype=numerical_dtype, device=device
    )
    velocity = torch.zeros_like(positions)
    config = projector_configuration(args.profile)
    soc_overrides = {
        name: value
        for name, value in (
            ("soc_admm_beta", args.soc_admm_beta),
            (
                "soc_admm_maximum_iterations",
                args.soc_admm_maximum_iterations,
            ),
            (
                "soc_admm_pcg_maximum_iterations",
                args.soc_admm_pcg_maximum_iterations,
            ),
            (
                "soc_admm_pcg_relative_tolerance",
                args.soc_admm_pcg_relative_tolerance,
            ),
            (
                "soc_admm_required_consecutive_gate_passes",
                args.soc_admm_required_consecutive_gate_passes,
            ),
        )
        if value is not None
    }
    if soc_overrides:
        if args.profile != "orientation_safe_soc_matrix_free":
            raise ValueError("SOC-ADMM CLI overrides require the SOC profile")
        config = replace(config, **soc_overrides)
        config.validate()
    projector = VolumetricMGPBDProjector(
        rest,
        mesh.elements,
        vertex_masses=np.ones(len(rest_np), dtype=np.float64),
        config=config,
    )
    rest_signed_six = signed_six_volumes(rest_np, mesh.elements)
    rest_triangles = rest_np.astype(np.float64)[mesh.boundary]
    rest_boundary_areas = 0.5 * np.linalg.norm(
        np.cross(
            rest_triangles[:, 1] - rest_triangles[:, 0],
            rest_triangles[:, 2] - rest_triangles[:, 0],
        ),
        axis=1,
    )
    if np.any(rest_boundary_areas <= 0.0):
        raise ValueError("reference boundary contains a degenerate rest face")

    initial_constraints = (
        projector.constraints_and_gradients(positions)[0].detach().cpu().numpy()
    )
    initial_velocity = np.zeros_like(squashed_np)
    initial_state = frame_state_metrics(
        squashed_np,
        initial_velocity,
        rest_np,
        mesh.elements,
        rest_signed_six,
        mesh.boundary,
        rest_boundary_areas,
        initial_constraints,
    )
    direct_oracle = (
        direct_first_direction_oracle(
            projector, positions, mesh.elements, rest_signed_six
        )
        if args.direct_linear_oracle
        else None
    )
    if direct_oracle is not None:
        (args.output / "direct_linear_oracle.json").write_text(
            json.dumps(direct_oracle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_boundary_obj(args.output / "rest_boundary.obj", rest_np, mesh.boundary)
    write_boundary_obj(
        args.output / "squashed_boundary.obj", squashed_np, mesh.boundary
    )

    frame_history: list[dict[str, object]] = []
    snapshot_frames = {1, 10, 50, 100}
    progress_path = args.output / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    run_started = time.perf_counter()
    for frame in range(1, args.frames + 1):
        old = positions.clone()
        predicted = positions if args.mode == "projection" else positions + config.dt_s * velocity
        synchronize(device)
        frame_started = time.perf_counter()
        try:
            positions = projector.project(predicted, post_iteration=None)
        except Exception as error:
            # An inner SOC convergence failure is a completed negative
            # numerical experiment, not missing evidence.  Preserve it as a
            # fail-closed GATE_FAILED run.  Infrastructure/programming errors
            # retain normal exception semantics and therefore produce FAILED.
            from .mgpbd_soc_admm import SOCADMMConvergenceError

            if not isinstance(error, SOCADMMConvergenceError):
                raise
            synchronize(device)
            current_np = (
                projector.last_accepted_outer_positions.detach().cpu().numpy()
            )
            accepted_outer_iterations = (
                projector.last_accepted_outer_iteration
            )
            failure_receipt = {
                "exception": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "inner_receipt": error.receipt,
                "projector_receipt": dict(projector.last_metrics),
                "last_safe_state": {
                    "scope": "atomically_accepted_nonlinear_outer_state",
                    "completed_outer_iterations": accepted_outer_iterations,
                    "frame_complete": False,
                },
            }
            (args.output / "soc_admm_failure_receipt.json").write_text(
                json.dumps(failure_receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            failure_record = {
                "frame": frame,
                "frame_ms": (time.perf_counter() - frame_started) * 1000.0,
                "projection_failed": True,
                "failure": failure_receipt,
            }
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure_record, sort_keys=True) + "\n")
                stream.flush()
            np.savez_compressed(
                args.output / "last_safe_state.npz",
                positions=current_np,
                velocity=velocity.detach().cpu().numpy(),
                boundary_faces=mesh.boundary,
                completed_outer_iterations=np.asarray(
                    accepted_outer_iterations, dtype=np.int64
                ),
            )
            write_boundary_obj(
                args.output / "last_safe_boundary.obj",
                current_np,
                mesh.boundary,
            )
            failure_payload = {
                "schema_version": (
                    "radeon_oneloop.mgpbd_bunny_conformance.v3"
                ),
                "formal": False,
                "physical_robot_output": False,
                "physical_leader_read": False,
                "hardware_output_enabled": False,
                "genesis_enabled": False,
                "contact_enabled": False,
                "claim_scope": {
                    "numerical_contract": args.contract,
                    "clean_room_implementation": True,
                    "complete_nonlinear_projector_loop_executed": False,
                    "isolated_direction_smoke": False,
                    "trajectory_claim": False,
                    "contact_claim": False,
                    "realtime_claim": False,
                    "hardware_claim": False,
                },
                "reference": {**mesh.diagnostics, "scene": scene},
                "configuration": {
                    "mode": args.mode,
                    "contract": args.contract,
                    "profile": args.profile,
                    "frames": args.frames,
                    "seed": args.seed,
                    "initial_height_ratio": args.initial_height_ratio,
                    "numerical_dtype": args.numerical_dtype,
                    "gravity_enabled": False,
                    "contact_enabled": False,
                    "integration_enabled": args.mode == "trajectory",
                    "projector": config.to_dict(),
                },
                "initial_state": initial_state,
                "failure": failure_receipt,
                "checks": {
                    "soc_admm_direction_completed": False,
                    "complete_nonlinear_projector_loop_completed": False,
                },
                "contract_valid": True,
                "quality_passed": False,
                "passed": False,
                "run_completed": True,
                "numerical_gate_completed": True,
            }
            (args.output / "metrics.json").write_text(
                json.dumps(failure_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(failure_payload, indent=2, sort_keys=True))
            return
        synchronize(device)
        frame_ms = (time.perf_counter() - frame_started) * 1000.0
        velocity = (
            torch.zeros_like(positions)
            if args.mode == "projection"
            else (positions - old) / config.dt_s
        )
        constraints = projector.constraints_and_gradients(positions)[0]
        synchronize(device)
        positions_np = positions.detach().cpu().numpy()
        velocity_np = velocity.detach().cpu().numpy()
        constraints_np = constraints.detach().cpu().numpy()
        state = frame_state_metrics(
            positions_np,
            velocity_np,
            rest_np,
            mesh.elements,
            rest_signed_six,
            mesh.boundary,
            rest_boundary_areas,
            constraints_np,
        )
        solver = dict(projector.last_metrics)
        frame_record = {
            "frame": frame,
            "frame_ms": frame_ms,
            "state": state,
            "solver": solver,
        }
        frame_history.append(frame_record)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(frame_record, sort_keys=True) + "\n")
            stream.flush()
        if frame in snapshot_frames or frame == args.frames:
            write_boundary_obj(
                args.output / f"frame_{frame:04d}_boundary.obj",
                positions_np,
                mesh.boundary,
            )
    synchronize(device)
    total_s = time.perf_counter() - run_started
    final_positions = positions.detach().cpu().numpy()
    final_velocity = velocity.detach().cpu().numpy()
    recovery = rigid_aligned_errors(final_positions, rest_np)
    rest_extents = np.ptp(rest_np.astype(np.float64), axis=0)
    final_extents = np.ptp(final_positions.astype(np.float64), axis=0)
    extent_relative_error = np.abs(final_extents / rest_extents - 1.0)
    initial_center = squashed_np.astype(np.float64).mean(axis=0)
    final_center = final_positions.astype(np.float64).mean(axis=0)
    bbox_diagonal = float(np.linalg.norm(rest_extents))
    center_drift_normalized = float(
        np.linalg.norm(final_center - initial_center) / bbox_diagonal
    )

    all_outer_records = [
        outer
        for frame in frame_history
        for outer in frame["solver"].get("outer_iterations", [])
    ]
    soc_contract = args.contract == "orientation_safe_soc_recovery"
    all_outer_soc_numerical_receipts = bool(all_outer_records) and all(
        soc_admm_numerical_receipt_consistent(record, config)
        for record in all_outer_records
    )
    all_outer_soc_transactions = bool(all_outer_records) and all(
        soc_admm_transaction_consistent(record)
        for record in all_outer_records
    )
    all_outer_soc_receipts_complete = bool(all_outer_records) and all(
        soc_admm_outer_receipt_consistent(record, config)
        for record in all_outer_records
    )
    linear_converged = (
        all_outer_soc_numerical_receipts
        if soc_contract
        else bool(all_outer_records)
        and all(
            float(record["pcg_relative_residual"])
            <= config.relative_residual * 1.05
            for record in all_outer_records
        )
    )
    outer_converged = all(
        float(frame["solver"]["outer_dual_final"])
        <= max(
            float(config.outer_absolute_residual),
            float(config.outer_relative_residual)
            * float(frame["solver"]["outer_dual_initial"]),
        )
        for frame in frame_history
    )
    correction_unclipped = bool(all_outer_records) and all(
        float(record["correction_global_scale"]) == 1.0
        for record in all_outer_records
    )
    all_outer_rap_current = bool(all_outer_records) and all(
        bool(record.get("amg_ready_for_solve"))
        and bool(record.get("rap_refreshed_for_current_matrix"))
        for record in all_outer_records
    )
    level0_matches_physical_operator = bool(all_outer_records) and all(
        float(record.get("level0_physical_operator_relative_error", np.inf))
        <= 5.0e-5
        for record in all_outer_records
    )
    all_finite = all(
        frame["state"]["finite_positions"] and frame["state"]["finite_velocity"]
        for frame in frame_history
    )
    all_positive = all(
        frame["state"]["inverted_tetrahedra"] == 0
        and frame["state"]["collapsed_tetrahedra"] == 0
        for frame in frame_history
    )
    topology = mesh.diagnostics["topology"]
    final_state = frame_history[-1]["state"]
    recovery_progress = arap_recovery_progress(initial_state, final_state)
    public_profile_and_builder = public_matrix_ua_profile_and_builder(
        args.profile, frame_history
    )
    builders = {
        str(frame["solver"].get("amg_hierarchy_builder"))
        for frame in frame_history
    }
    expected_builder = (
        "not_built"
        if soc_contract
        else (
            "static_tet_topology_greedy_UA"
            if args.profile == "topology_ua_ablation"
            else "PyAMG_plain_UA_smooth_none_clean_room"
        )
    )
    profile_hierarchy_consistent = builders == {expected_builder}
    soc_legacy_direction_path_absent = bool(all_outer_records) and all(
        record.get("direction_backend") == "soc_admm"
        and record.get("legacy_direction_pcg_skipped") is True
        and int(record.get("pcg_iterations", -1)) == 0
        and record.get("amg_ready_for_solve") is False
        and record.get("rap_refreshed_for_current_matrix") is False
        for record in all_outer_records
    ) and all(
        int(frame["solver"].get("pcg_iterations_total", -1)) == 0
        and int(frame["solver"].get("amg_rap_numeric_refreshes_last_project", -1))
        == 0
        and frame["solver"].get("amg_hierarchy_builder") == "not_built"
        for frame in frame_history
    )
    public_line_search_metrics_consistent = bool(all_outer_records) and all(
        public_line_search_record_consistent(record)
        for record in all_outer_records
    )
    all_outer_atomic = bool(all_outer_records) and all(
        bool(record.get("position_and_lagrangian_step_accepted_atomically"))
        and multiplier_contract_consistent(record, rejected_policy="rollback")
        for record in all_outer_records
    )
    all_outer_sqp_scaled_transactions = bool(all_outer_records) and all(
        scaled_sqp_multiplier_contract_consistent(record)
        for record in all_outer_records
    )
    all_outer_sqp_armijo_atomic = bool(all_outer_records) and all(
        sqp_armijo_and_atomic_receipt_consistent(record, config)
        for record in all_outer_records
    )
    all_outer_sqp_numerical_receipts = bool(all_outer_records) and all(
        sqp_numerical_receipt_consistent(record, config)
        for record in all_outer_records
    )
    all_outer_sqp_receipts_complete = bool(all_outer_records) and all(
        sqp_receipt_contract_consistent(record, config)
        for record in all_outer_records
    )
    all_outer_accepted_positive = bool(all_outer_records) and all(
        float(record.get("accepted_minimum_signed_volume_ratio", -np.inf))
        >= config.orientation_guard_minimum_ratio
        and int(record.get("accepted_inverted_tetrahedra", 1)) == 0
        and int(record.get("accepted_collapsed_tetrahedra", 1)) == 0
        for record in all_outer_records
    )
    all_outer_within_strain_trust_region = bool(all_outer_records) and all(
        bool(record.get("strain_trust_filter_is_downstream_extension"))
        and np.isfinite(
            float(record.get("accepted_maximum_arap_strain", np.nan))
        )
        and float(record.get("accepted_maximum_arap_strain", np.inf))
        <= config.strain_trust_filter_maximum * (1.0 + 1.0e-6)
        for record in all_outer_records
    )
    exact_initialization = (
        float(np.ptp(squashed_np[:, 1])) == 0.0
        and np.array_equal(squashed_np[:, (0, 2)], rest_np[:, (0, 2)])
    )
    positive_initialization = (
        np.array_equal(squashed_np[:, (0, 2)], rest_np[:, (0, 2)])
        and np.isclose(
            float(np.ptp(squashed_np[:, 1]) / np.ptp(rest_np[:, 1])),
            args.initial_height_ratio,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        and int(initial_state["inverted_tetrahedra"]) == 0
        and int(initial_state["collapsed_tetrahedra"]) == 0
    )
    forbidden_runtime_modules_absent = not any(
        name.startswith(
            (
                "genesis",
                "serial",
                "sim.genesis_so101.live_teleop",
                "sim.genesis_so101.scene",
            )
        )
        for name in sys.modules
    )
    contract_checks = {
        "pinned_reference_commit": mesh.diagnostics["commit"] == REFERENCE_COMMIT,
        "pinned_scene_verified": scene["sha256"] is not None,
        "closed_genus_zero_boundary": (
            topology["edge_incidence_not_two"] == 0
            and topology["connected_components"] == 1
            and topology["genus"] == 0.0
        ),
        "source_orientation_normalized": (
            mesh.diagnostics["orientation"]["normalized_nonpositive_tetrahedra"]
            == 0
        ),
        "contract_initialization_matches_label": (
            exact_initialization
            if args.contract in {"official_fidelity", "numerical_ablation"}
            else positive_initialization
        ),
        "uniform_reference_vertex_mass": projector.mass_model == "explicit_per_vertex",
        "profile_hierarchy_consistent": profile_hierarchy_consistent,
        "official_fidelity_profile_eligible_when_claimed": (
            args.contract != "official_fidelity" or public_profile_and_builder
        ),
        "official_line_search_semantics_when_claimed": (
            args.contract != "official_fidelity"
            or (
                config.line_search_enabled
                and config.line_search_objective == "dual"
                and config.line_search_acceptance_epsilon == 0.0
                and config.line_search_minimum_step == 1.0e-9
                and not config.line_search_scale_lagrangian
                and config.line_search_rejected_lagrangian_policy
                == "retain_full"
                and public_line_search_metrics_consistent
            )
        ),
        "orientation_safe_contract_enabled_when_claimed": (
            args.contract != "orientation_safe_recovery"
            or (
                config.orientation_guard_enabled
                and config.orientation_diagnostics_enabled
                and config.line_search_rejected_lagrangian_policy
                == "rollback"
                and config.strain_trust_filter_enabled
                and config.strain_trust_filter_maximum == 1.0
                and float(initial_state["arap_maximum"])
                <= config.strain_trust_filter_maximum
            )
        ),
        "orientation_safe_sqp_contract_enabled_when_claimed": (
            args.contract != "orientation_safe_sqp_recovery"
            or (
                config.sqp_direction_enabled
                and config.pcg_iterations == 1_000
                and config.maximum_correction_m is None
                and config.amg_hierarchy_mode == "matrix_ua"
                and config.line_search_enabled
                and config.line_search_objective == "dual"
                and config.line_search_scale_lagrangian
                and config.line_search_rejected_lagrangian_policy
                == "rollback"
                and config.orientation_guard_enabled
                and config.orientation_diagnostics_enabled
                and config.strain_trust_filter_enabled
                and config.strain_trust_filter_maximum == 1.0
                and config.sqp_strain_maximum
                == config.strain_trust_filter_maximum
                and np.isclose(
                    args.initial_height_ratio, 0.25, rtol=0.0, atol=1.0e-12
                )
                and float(initial_state["arap_maximum"])
                <= config.sqp_strain_maximum
            )
        ),
        "orientation_safe_soc_contract_enabled_when_claimed": (
            not soc_contract
            or (
                config.soc_admm_direction_enabled
                and not config.sqp_direction_enabled
                and config.maximum_correction_m is None
                and config.line_search_enabled
                and config.line_search_objective == "dual"
                and config.line_search_scale_lagrangian
                and config.line_search_rejected_lagrangian_policy
                == "rollback"
                and config.orientation_guard_enabled
                and config.orientation_diagnostics_enabled
                and config.strain_trust_filter_enabled
                # The accepted outer state must itself be a feasible zero
                # direction for the next SOC subproblem.  Its trust limit is
                # therefore bounded by the *work* radius (0.989 in P0a2), not
                # by the looser accepted-candidate ARAP audit maximum (1.0).
                and config.strain_trust_filter_maximum
                <= config.soc_admm_work_radius
                and config.soc_admm_work_radius
                < config.soc_admm_true_arap_maximum
                and config.soc_admm_minimum_signed_volume_ratio
                >= config.orientation_guard_minimum_ratio
                and config.soc_admm_work_radius
                < min(
                    config.soc_admm_true_arap_maximum,
                    1.0
                    - config.soc_admm_minimum_signed_volume_ratio
                    ** (1.0 / 3.0),
                )
                and config.soc_admm_primal_tolerance
                <= 0.5
                * (
                    min(
                        config.soc_admm_true_arap_maximum,
                        1.0
                        - config.soc_admm_minimum_signed_volume_ratio
                        ** (1.0 / 3.0),
                    )
                    - config.soc_admm_work_radius
                )
                and config.soc_admm_maximum_iterations >= 2_000
                and config.soc_admm_pcg_maximum_iterations >= 2_000
                and config.soc_admm_pcg_relative_tolerance <= 1.5e-5
                and config.soc_admm_required_consecutive_gate_passes >= 2
                and np.isclose(
                    args.initial_height_ratio, 0.25, rtol=0.0, atol=1.0e-12
                )
                and float(initial_state["arap_maximum"])
                <= config.soc_admm_work_radius
            )
        ),
        "legacy_hierarchy_current_when_used": (
            soc_contract or all_outer_rap_current
        ),
        "legacy_level0_matrix_matches_physical_operator_when_used": (
            soc_contract or level0_matches_physical_operator
        ),
        "soc_bypasses_legacy_direction_pcg_and_hierarchy_when_claimed": (
            not soc_contract or soc_legacy_direction_path_absent
        ),
        "live_correction_cap_disabled": (
            config.maximum_correction_m is None and correction_unclipped
        ),
        "all_states_finite": all_finite,
        "forbidden_runtime_modules_absent": forbidden_runtime_modules_absent,
    }
    quality_checks = {
        "all_inner_linear_solves_converged": linear_converged,
        "all_outer_dual_solves_converged": outer_converged,
        "mass_center_conserved": center_drift_normalized <= 1.0e-4,
        "final_boundary_faces_nondegenerate": (
            final_state["boundary"]["degenerate_faces"] == 0
        ),
    }
    if args.contract == "orientation_safe_recovery":
        quality_checks.update(
            {
                "all_post_projection_tets_positive": all_positive,
                "every_outer_accepted_state_positive": (
                    all_outer_accepted_positive
                ),
                "position_and_lagrangian_acceptance_atomic": all_outer_atomic,
                "every_outer_state_within_strain_trust_region": (
                    all_outer_within_strain_trust_region
                ),
                "all_frame_states_within_strain_trust_region": all(
                    float(frame["state"]["arap_maximum"])
                    <= config.strain_trust_filter_maximum * (1.0 + 1.0e-6)
                    for frame in frame_history
                ),
                "arap_norm_maximum_and_y_extent_recovery_progress": (
                    recovery_progress
                ),
                "trajectory_shape_recovery": (
                    args.mode != "trajectory"
                    or (
                        recovery["rms_over_rest_bbox_diagonal"] <= 0.02
                        and float(np.max(extent_relative_error)) <= 0.05
                    )
                ),
            }
        )
    if args.contract == "orientation_safe_sqp_recovery":
        quality_checks.update(
            {
                "all_post_projection_tets_positive": all_positive,
                "every_outer_accepted_state_positive": (
                    all_outer_accepted_positive
                ),
                "every_outer_scaled_position_multiplier_transaction": (
                    all_outer_sqp_scaled_transactions
                ),
                "every_outer_sqp_armijo_or_atomic_rollback": (
                    all_outer_sqp_armijo_atomic
                ),
                "every_outer_sqp_kkt_auxiliary_and_coupled_residuals_pass": (
                    all_outer_sqp_numerical_receipts
                ),
                "every_outer_sqp_receipt_complete": (
                    all_outer_sqp_receipts_complete
                ),
                "every_outer_state_within_strain_trust_region": (
                    all_outer_within_strain_trust_region
                ),
                "all_frame_states_within_strain_trust_region": all(
                    float(frame["state"]["arap_maximum"])
                    <= config.strain_trust_filter_maximum * (1.0 + 1.0e-6)
                    for frame in frame_history
                ),
                "arap_norm_maximum_and_y_extent_recovery_progress": (
                    recovery_progress
                ),
                "trajectory_shape_recovery": (
                    args.mode != "trajectory"
                    or (
                        recovery["rms_over_rest_bbox_diagonal"] <= 0.02
                        and float(np.max(extent_relative_error)) <= 0.05
                    )
                ),
            }
        )
    if soc_contract:
        quality_checks.update(
            {
                "all_post_projection_tets_positive": all_positive,
                "every_outer_accepted_state_positive": (
                    all_outer_accepted_positive
                ),
                "every_outer_soc_admm_numerical_receipt_passes": (
                    all_outer_soc_numerical_receipts
                ),
                "every_outer_soc_position_multiplier_transaction_atomic": (
                    all_outer_soc_transactions
                ),
                "every_outer_soc_receipt_complete": (
                    all_outer_soc_receipts_complete
                ),
                "every_outer_state_within_strain_trust_region": (
                    all_outer_within_strain_trust_region
                ),
                "all_frame_states_within_strain_trust_region": all(
                    float(frame["state"]["arap_maximum"])
                    <= config.strain_trust_filter_maximum * (1.0 + 1.0e-6)
                    for frame in frame_history
                ),
                "arap_norm_maximum_and_y_extent_recovery_progress": (
                    recovery_progress
                ),
                "trajectory_shape_recovery": (
                    args.mode != "trajectory"
                    or (
                        recovery["rms_over_rest_bbox_diagonal"] <= 0.02
                        and float(np.max(extent_relative_error)) <= 0.05
                    )
                ),
            }
        )
    checks = {**contract_checks, **quality_checks}
    step_times = np.asarray(
        [float(frame["frame_ms"]) for frame in frame_history], dtype=np.float64
    )
    contract_valid = all(contract_checks.values())
    quality_passed = all(quality_checks.values())
    payload = {
        "schema_version": "radeon_oneloop.mgpbd_bunny_conformance.v3",
        "formal": False,
        "physical_robot_output": False,
        "physical_leader_read": False,
        "hardware_output_enabled": False,
        "genesis_enabled": False,
        "contact_enabled": False,
        "claim_scope": {
            "numerical_contract": args.contract,
            "clean_room_implementation": True,
            "official_binary_executed": False,
            "official_trajectory_oracle_available": False,
            "official_trajectory_parity_passed": False,
            "material_unit_calibration": False,
            "zero_inversion_is_official_contract": False,
            "orientation_safe_is_downstream_extension": (
                args.contract
                in {
                    "orientation_safe_recovery",
                    "orientation_safe_sqp_recovery",
                    "orientation_safe_soc_recovery",
                }
            ),
            "strain_trust_filter_is_downstream_extension": (
                args.contract
                in {
                    "orientation_safe_recovery",
                    "orientation_safe_sqp_recovery",
                    "orientation_safe_soc_recovery",
                }
                and config.strain_trust_filter_enabled
            ),
            "strain_trust_filter_is_official_contract": False,
            "active_set_sqp_is_downstream_extension": (
                args.contract == "orientation_safe_sqp_recovery"
                and config.sqp_direction_enabled
            ),
            "active_set_sqp_is_official_contract": False,
            "soc_admm_is_downstream_extension": (
                soc_contract and config.soc_admm_direction_enabled
            ),
            "soc_admm_is_official_contract": False,
            "complete_nonlinear_projector_loop_executed": soc_contract,
            "isolated_direction_smoke": False,
            "genesis_claim": False,
            "contact_claim": False,
            "hardware_claim": False,
            "scaled_position_multiplier_transaction": (
                args.contract
                in {
                    "orientation_safe_sqp_recovery",
                    "orientation_safe_soc_recovery",
                }
                and config.line_search_scale_lagrangian
            ),
            "bunny_small_uses_full_scene_parameters_as_kernel_smoke": (
                args.model == "bunny_small"
            ),
        },
        "reference": {**mesh.diagnostics, "scene": scene},
        "configuration": {
            "mode": args.mode,
            "contract": args.contract,
            "profile": args.profile,
            "frames": args.frames,
            "seed": args.seed,
            "initial_height_ratio": (
                0.0
                if args.contract in {"official_fidelity", "numerical_ablation"}
                else args.initial_height_ratio
            ),
            "numerical_dtype": args.numerical_dtype,
            "direct_linear_oracle_enabled": args.direct_linear_oracle,
            "mass_model": "uniform_per_vertex",
            "mass_per_vertex_sim_units": 1.0,
            "gravity_enabled": False,
            "contact_enabled": False,
            "integration_enabled": args.mode == "trajectory",
            "projector": config.to_dict(),
        },
        "initial_state": initial_state,
        "direct_linear_oracle": direct_oracle,
        "final_state": final_state,
        "recovery": {
            **recovery,
            "extent_relative_error": extent_relative_error.tolist(),
            "center_drift_over_rest_bbox_diagonal": center_drift_normalized,
        },
        "performance": {
            "total_s": total_s,
            "frame_ms_mean": float(np.mean(step_times)),
            "frame_ms_p50": float(np.percentile(step_times, 50)),
            "frame_ms_p95": float(np.percentile(step_times, 95)),
            "frame_ms_maximum": float(np.max(step_times)),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "convergence": {
            "frames": frame_history,
            "all_inner_linear_solves_converged": linear_converged,
            "all_outer_dual_solves_converged": outer_converged,
            "level0_matches_physical_operator": (
                level0_matches_physical_operator
            ),
            "all_outer_sqp_scaled_transactions": (
                all_outer_sqp_scaled_transactions
            ),
            "all_outer_sqp_armijo_atomic": all_outer_sqp_armijo_atomic,
            "all_outer_sqp_numerical_receipts": (
                all_outer_sqp_numerical_receipts
            ),
            "all_outer_sqp_receipts_complete": (
                all_outer_sqp_receipts_complete
            ),
            "all_outer_soc_numerical_receipts": (
                all_outer_soc_numerical_receipts
            ),
            "all_outer_soc_transactions": all_outer_soc_transactions,
            "all_outer_soc_receipts_complete": (
                all_outer_soc_receipts_complete
            ),
        },
        "forbidden_features": {
            "genesis": False,
            "post_iteration_contact": False,
            "coarse_transport": False,
            "center_lock": False,
            "jaw_limiter": False,
            "synthetic_attachment": False,
            "separate_visual_mesh": False,
        },
        "observations_not_universal_gates": {
            "all_post_projection_tets_positive": all_positive,
            "arap_norm_and_maximum_recovered": recovery_progress,
            "official_fidelity_eligible": public_profile_and_builder,
        },
        "contract_checks": contract_checks,
        "quality_checks": quality_checks,
        "checks": checks,
        "contract_valid": contract_valid,
        "quality_passed": quality_passed,
        "passed": contract_valid and quality_passed,
        "run_completed": True,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "residual_history.jsonl").write_text(
        "".join(json.dumps(frame, sort_keys=True) + "\n" for frame in frame_history),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output / "final_state.npz",
        positions=final_positions,
        velocity=final_velocity,
        boundary_faces=mesh.boundary,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
