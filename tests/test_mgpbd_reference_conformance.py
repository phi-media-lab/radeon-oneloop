import numpy as np
import pytest

from sim.genesis_so101.mgpbd_bunny_conformance import (
    arap_recovery_progress,
    scaled_sqp_multiplier_contract_consistent,
    positive_height_squash,
    projector_configuration,
    public_matrix_ua_profile_and_builder,
    public_line_search_record_consistent,
    multiplier_contract_consistent,
    rigid_aligned_errors,
    soc_admm_numerical_receipt_consistent,
    soc_admm_outer_receipt_consistent,
    soc_admm_transaction_consistent,
    sqp_armijo_and_atomic_receipt_consistent,
    sqp_numerical_receipt_consistent,
    sqp_receipt_contract_consistent,
    squash_y,
)
from sim.genesis_so101.mgpbd_reference_io import (
    boundary_topology,
    normalize_consistent_orientation,
    read_tetgen_elements,
    read_tetgen_nodes,
    signed_six_volumes,
)
from sim.genesis_so101.mgpbd_tet import (
    MGPBDTetConfig,
    VolumetricMGPBDProjector,
    resolve_vertex_masses,
)


def test_tetgen_parser_maps_noncontiguous_node_identifiers(tmp_path) -> None:
    node = tmp_path / "fixture.node"
    element = tmp_path / "fixture.ele"
    node.write_text(
        "4 3 0 0\n"
        "10 0 0 0\n"
        "20 1 0 0\n"
        "30 0 1 0\n"
        "40 0 0 1\n",
        encoding="utf-8",
    )
    element.write_text("1 4 0\n7 10 30 20 40\n", encoding="utf-8")
    identifiers, positions = read_tetgen_nodes(node)
    element_ids, elements = read_tetgen_elements(element, identifiers)
    assert identifiers.tolist() == [10, 20, 30, 40]
    assert element_ids.tolist() == [7]
    assert positions.shape == (4, 3)
    assert elements.tolist() == [[0, 2, 1, 3]]


def test_reference_orientation_normalizes_consistent_negative_tets() -> None:
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    negative = np.asarray(((0, 2, 1, 3),), dtype=np.int32)
    normalized, diagnostics = normalize_consistent_orientation(positions, negative)
    assert diagnostics["source_negative_tetrahedra"] == 1
    assert diagnostics["normalization"] == "swap_local_1_2"
    assert signed_six_volumes(positions, normalized)[0] > 0.0


def test_tetrahedron_boundary_is_closed_genus_zero() -> None:
    faces = np.asarray(
        ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)), dtype=np.int32
    )
    topology = boundary_topology(faces)
    assert topology["boundary_vertices"] == 4
    assert topology["boundary_faces"] == 4
    assert topology["boundary_edges"] == 6
    assert topology["edge_incidence_not_two"] == 0
    assert topology["connected_components"] == 1
    assert topology["genus"] == pytest.approx(0.0)


def test_uniform_reference_mass_does_not_change_live_mass_default() -> None:
    elements = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    volumes = np.asarray((1.0 / 6.0,))
    explicit, explicit_model = resolve_vertex_masses(
        elements,
        volumes,
        4,
        total_mass=None,
        explicit_vertex_masses=np.ones(4),
    )
    lumped, lumped_model = resolve_vertex_masses(
        elements,
        volumes,
        4,
        total_mass=0.04,
    )
    np.testing.assert_array_equal(explicit, np.ones(4))
    assert explicit_model == "explicit_per_vertex"
    assert lumped.sum() == pytest.approx(0.04)
    assert lumped_model == "volume_lumped_total_mass"
    assert MGPBDTetConfig().maximum_correction_m == pytest.approx(0.012)


def test_public_profile_isolated_from_live_contact_defaults() -> None:
    config = projector_configuration("public_matrix_ua")
    config.validate()
    assert config.dt_s == pytest.approx(0.01)
    assert config.shear_modulus_pa == pytest.approx(1.0e9)
    assert config.nonlinear_iterations == 20
    assert config.pcg_iterations == 100
    assert config.relative_residual == pytest.approx(1.0e-5)
    assert config.outer_absolute_residual == pytest.approx(1.0e-4)
    assert config.outer_relative_residual == pytest.approx(1.0e-2)
    assert config.maximum_correction_m is None
    assert config.amg_hierarchy_mode == "matrix_ua"
    assert not config.symmetric_diagonal_equilibration
    assert config.line_search_enabled
    assert config.line_search_objective == "dual"
    assert config.line_search_acceptance_epsilon == 0.0
    assert config.line_search_minimum_step == pytest.approx(1.0e-9)
    assert not config.line_search_scale_lagrangian
    assert config.line_search_rejected_lagrangian_policy == "retain_full"
    assert config.orientation_diagnostics_enabled
    assert not config.orientation_guard_enabled
    assert not config.sqp_direction_enabled
    assert config.relaxation == pytest.approx(1.0)


def test_ablation_profiles_are_not_the_public_profile() -> None:
    for profile in (
        "paper_fixed_omega",
        "radeon_equilibrated_matrix_ua",
        "topology_ua_ablation",
    ):
        assert projector_configuration(profile) != projector_configuration(
            "public_matrix_ua"
        )


def test_public_builder_gate_rejects_ablation_labels() -> None:
    frames = [
        {
            "solver": {
                "amg_hierarchy_builder": (
                    "PyAMG_plain_UA_smooth_none_clean_room"
                )
            }
        }
    ]
    assert public_matrix_ua_profile_and_builder("public_matrix_ua", frames)
    assert not public_matrix_ua_profile_and_builder(
        "radeon_equilibrated_matrix_ua", frames
    )
    assert not public_matrix_ua_profile_and_builder("topology_ua_ablation", frames)


def test_recovery_gate_requires_worst_tet_strain_to_decrease() -> None:
    initial = {
        "arap_l2": 100.0,
        "arap_maximum": 1.0,
        "extent_ratio_to_rest": [1.0, 0.0, 1.0],
    }
    misleading_final = {
        "arap_l2": 20.0,
        "arap_maximum": 4.0,
        "extent_ratio_to_rest": [1.0, 0.7, 1.0],
    }
    coherent_final = {
        "arap_l2": 20.0,
        "arap_maximum": 0.5,
        "extent_ratio_to_rest": [1.0, 0.7, 1.0],
    }
    assert not arap_recovery_progress(initial, misleading_final)
    assert arap_recovery_progress(initial, coherent_final)


def test_squash_and_rigid_alignment_metrics() -> None:
    rest = np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 2.0, 0.0), (0.0, 1.0, 1.0)), dtype=np.float32
    )
    squashed = squash_y(rest)
    assert np.ptp(squashed[:, 1]) == 0.0
    np.testing.assert_array_equal(squashed[:, (0, 2)], rest[:, (0, 2)])
    translated = rest + np.asarray((4.0, -3.0, 2.0), dtype=np.float32)
    aligned = rigid_aligned_errors(translated, rest)
    assert aligned["rms_over_rest_bbox_diagonal"] == pytest.approx(0.0, abs=1.0e-7)


def test_positive_height_squash_retains_affine_orientation() -> None:
    rest = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    compressed = positive_height_squash(rest, 0.1)
    ratio = signed_six_volumes(compressed, np.asarray(((0, 1, 2, 3),)))
    ratio /= signed_six_volumes(rest, np.asarray(((0, 1, 2, 3),)))
    assert ratio.tolist() == pytest.approx((0.1,))


def test_orientation_safe_profile_is_strict_and_coupled() -> None:
    config = projector_configuration("orientation_safe_matrix_ua")
    config.validate()
    assert config.pcg_iterations == 1_000
    assert config.nonlinear_iterations == 60
    assert config.orientation_diagnostics_enabled
    assert config.orientation_guard_enabled
    assert not config.line_search_scale_lagrangian
    assert config.line_search_rejected_lagrangian_policy == "rollback"
    assert config.strain_trust_filter_enabled
    assert config.strain_trust_filter_maximum == pytest.approx(1.0)
    assert config.line_search_acceptance_epsilon == 0.0
    assert config.line_search_minimum_step == pytest.approx(1.0e-9)
    assert not config.sqp_direction_enabled


def test_orientation_safe_sqp_profile_is_strict_scaled_and_unclipped() -> None:
    config = projector_configuration("orientation_safe_sqp_matrix_ua")
    config.validate()
    assert config.pcg_iterations == 1_000
    assert config.nonlinear_iterations == 60
    assert config.maximum_correction_m is None
    assert config.amg_hierarchy_mode == "matrix_ua"
    assert not config.symmetric_diagonal_equilibration
    assert config.orientation_diagnostics_enabled
    assert config.orientation_guard_enabled
    assert config.strain_trust_filter_enabled
    assert config.strain_trust_filter_maximum == pytest.approx(1.0)
    assert config.sqp_direction_enabled
    assert config.sqp_strain_maximum == pytest.approx(1.0)
    assert config.line_search_enabled
    assert config.line_search_objective == "dual"
    assert config.line_search_scale_lagrangian
    assert config.line_search_rejected_lagrangian_policy == "rollback"
    assert config.line_search_acceptance_epsilon == 0.0
    assert config.line_search_minimum_step == pytest.approx(1.0e-9)


def test_orientation_safe_soc_profile_runs_full_projector_not_legacy_pcg() -> None:
    config = projector_configuration("orientation_safe_soc_matrix_free")
    config.validate()
    assert config.nonlinear_iterations == 60
    assert config.soc_admm_direction_enabled
    assert not config.sqp_direction_enabled
    assert config.maximum_correction_m is None
    assert config.line_search_enabled
    assert config.line_search_objective == "dual"
    assert config.line_search_scale_lagrangian
    assert config.line_search_rejected_lagrangian_policy == "rollback"
    assert config.orientation_guard_enabled
    assert config.strain_trust_filter_enabled
    assert config.strain_trust_filter_maximum == pytest.approx(0.989)
    assert config.strain_trust_filter_maximum <= config.soc_admm_work_radius
    assert config.soc_admm_work_radius < config.soc_admm_true_arap_maximum
    assert config.soc_admm_beta == pytest.approx(1.0e-4)
    assert config.soc_admm_beta_maximum == pytest.approx(1.28e-2)
    assert config.soc_admm_kkt_polish_beta_maximum is None
    assert config.soc_admm_maximum_iterations == 2_000
    assert config.soc_admm_pcg_maximum_iterations == 2_000
    assert config.soc_admm_pcg_relative_tolerance == pytest.approx(1.5e-5)
    assert config.soc_admm_required_consecutive_gate_passes == 2


def _complete_soc_record(*, rejected: bool = False) -> dict[str, object]:
    config = projector_configuration("orientation_safe_soc_matrix_free")
    step = 0.0 if rejected else 0.25
    policy = (
        "rollback_multiplier_with_position"
        if rejected
        else "accepted_scaled_trial_multiplier"
    )
    merit_before = 2.0
    merit_slope = -4.0
    coefficient = 1.0e-4
    proof_radius = min(
        config.soc_admm_true_arap_maximum,
        1.0
        - config.soc_admm_minimum_signed_volume_ratio ** (1.0 / 3.0),
    )
    checks = {
        name: True
        for name in (
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
        )
    }
    receipt = {
        "backend": "Torch_ROCm_matrix_free_SOC_ADMM",
        "converged": True,
        "fallback_used": False,
        "passed": True,
        "failure": None,
        "configuration": {
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
            "pcg_relative_tolerance": (
                config.soc_admm_pcg_relative_tolerance
            ),
            "proof_radius": proof_radius,
        },
        "checks": checks,
        "precision_continuation_active": False,
        "precision_continuation_started_iteration": None,
        "precision_continuation_events": [],
        "admm_iterations": 10,
        "pcg_solves": 1,
        "pcg_iterations_total": 7,
        "pcg_receipts": [
            {"converged": True, "true_residual_to_target": 0.5}
        ],
        "consecutive_gate_passes_final": 2,
        "adaptive_beta_update_count": 0,
        "admm_primal_residual_maximum": 1.0e-5,
        "admm_dual_residual_relative": 1.0e-5,
        "stationarity_relative": 1.0e-5,
        "normal_cone": {
            "dual_convention": "physical_y_equals_beta_times_scaled_dual",
            "gate_residual": 1.0e-6,
            "maximum_projection_fixed_point_residual": 1.0e-6
        },
        "safety_proof_radius_maximum": 0.98,
        "true_arap_maximum": 0.8,
        "minimum_signed_volume_ratio": 0.1,
        "inverted_or_collapsed_tetrahedra": 0,
        "coupled_material_residual_relative": 1.0e-7,
        "soc_z_violation_maximum": 0.0,
        "initial_objective": 2.0,
        "final_objective": 1.0,
        "direction_l2": 1.0,
        "delta_lambda_l2": 1.0,
        "merit_slope": merit_slope,
        "coupled_position_multiplier_transaction": True,
        "accepted_step": step,
        "accepted_multiplier_fraction": step,
        "multiplier_acceptance_policy": policy,
        "armijo_coefficient": coefficient,
        "armijo_merit_before": merit_before,
        "armijo_merit_after": merit_before if rejected else 1.5,
        "armijo_rhs": merit_before + coefficient * step * merit_slope,
        "armijo_satisfied": not rejected,
        "rolled_back_atomically": True,
    }
    return {
        "direction_backend": "soc_admm",
        "legacy_direction_pcg_skipped": True,
        "line_search_rejected": rejected,
        "line_search_step": step,
        "correction_global_scale": 1.0,
        "position_and_lagrangian_step_accepted_atomically": True,
        "lagrangian_acceptance_policy": policy,
        "lagrangian_update_fraction": step,
        "lagrangian_fraction_matches_observed": True,
        "lagrangian_transaction_relative_error": 0.0,
        "soc_admm_direction": receipt,
    }


def test_soc_outer_gate_requires_numerics_transaction_and_raw_receipt() -> None:
    config = projector_configuration("orientation_safe_soc_matrix_free")
    accepted = _complete_soc_record()
    rejected = _complete_soc_record(rejected=True)
    assert soc_admm_numerical_receipt_consistent(accepted, config)
    assert soc_admm_transaction_consistent(accepted)
    assert soc_admm_outer_receipt_consistent(accepted, config)
    assert soc_admm_outer_receipt_consistent(rejected, config)

    missing_pcg = _complete_soc_record()
    del missing_pcg["soc_admm_direction"]["pcg_receipts"]
    assert not soc_admm_numerical_receipt_consistent(missing_pcg, config)
    unsafe = _complete_soc_record()
    unsafe["soc_admm_direction"]["minimum_signed_volume_ratio"] = 0.0
    assert not soc_admm_numerical_receipt_consistent(unsafe, config)
    missing_precision_contract = _complete_soc_record()
    del missing_precision_contract["soc_admm_direction"][
        "precision_continuation_active"
    ]
    assert not soc_admm_numerical_receipt_consistent(
        missing_precision_contract, config
    )
    incomplete_continuation = _complete_soc_record()
    continuation_receipt = incomplete_continuation["soc_admm_direction"]
    continuation_receipt["precision_continuation_active"] = True
    continuation_receipt["precision_continuation_started_iteration"] = 8
    continuation_receipt["precision_continuation_events"] = [
        {"admm_iteration": 8, "reason": "float32_true_residual_floor"}
    ]
    assert not soc_admm_numerical_receipt_consistent(
        incomplete_continuation, config
    )
    complete_continuation = _complete_soc_record()
    continuation_receipt = complete_continuation["soc_admm_direction"]
    continuation_receipt["precision_continuation_active"] = True
    continuation_receipt["precision_continuation_started_iteration"] = 8
    continuation_receipt["precision_continuation_events"] = [
        {"admm_iteration": 8, "reason": "float32_true_residual_floor"}
    ]
    continuation_receipt["accepted_dtype_reaudit"] = {
        "dtype": "torch.float32",
        "checks": {"finite_candidate": True, "stationarity_satisfied": True},
        "passed": True,
    }
    continuation_receipt["checks"]["accepted_dtype_reaudit_satisfied"] = True
    assert soc_admm_numerical_receipt_consistent(
        complete_continuation, config
    )
    failed_reaudit = _complete_soc_record()
    continuation_receipt = failed_reaudit["soc_admm_direction"]
    continuation_receipt["precision_continuation_active"] = True
    continuation_receipt["precision_continuation_started_iteration"] = 8
    continuation_receipt["precision_continuation_events"] = [
        {"admm_iteration": 8, "reason": "float32_true_residual_floor"}
    ]
    continuation_receipt["accepted_dtype_reaudit"] = {
        "dtype": "torch.float32",
        "checks": {"stationarity_satisfied": False},
        "passed": False,
    }
    continuation_receipt["checks"]["accepted_dtype_reaudit_satisfied"] = False
    assert not soc_admm_numerical_receipt_consistent(failed_reaudit, config)
    stationary = _complete_soc_record()
    stationary_receipt = stationary["soc_admm_direction"]
    stationary_receipt["merit_slope"] = 0.0
    stationary_receipt["armijo_merit_before"] = 0.0
    stationary_receipt["armijo_merit_after"] = 0.0
    stationary_receipt["armijo_rhs"] = 0.0
    stationary_receipt["direction_l2"] = 0.0
    stationary_receipt["delta_lambda_l2"] = 0.0
    stationary_receipt["initial_objective"] = 0.0
    stationary_receipt["final_objective"] = 0.0
    assert soc_admm_transaction_consistent(stationary)
    assert soc_admm_outer_receipt_consistent(stationary, config)
    wrong_backend = _complete_soc_record()
    wrong_backend["legacy_direction_pcg_skipped"] = False
    assert not soc_admm_transaction_consistent(wrong_backend)
    mismatched_fraction = _complete_soc_record()
    mismatched_fraction["lagrangian_update_fraction"] = 1.0
    assert not soc_admm_transaction_consistent(mismatched_fraction)


def test_multiplier_contract_gate_rejects_false_atomic_claims() -> None:
    accepted = {
        "line_search_rejected": False,
        "lagrangian_acceptance_policy": "accepted_full_trial_multiplier",
        "lagrangian_update_fraction": 1.0,
        "lagrangian_fraction_matches_observed": True,
        "lagrangian_transaction_relative_error": 0.0,
    }
    rolled_back = {
        "line_search_rejected": True,
        "lagrangian_acceptance_policy": "rollback_multiplier_with_position",
        "lagrangian_update_fraction": 0.0,
        "lagrangian_fraction_matches_observed": True,
        "lagrangian_transaction_relative_error": 0.0,
    }
    false_pass = {
        **rolled_back,
        "lagrangian_update_fraction": 1.0,
    }
    retained = {
        "line_search_rejected": True,
        "lagrangian_acceptance_policy": "retain_full_public_multiplier",
        "lagrangian_update_fraction": 1.0,
        "lagrangian_fraction_matches_observed": True,
        "lagrangian_transaction_relative_error": 0.0,
    }
    assert multiplier_contract_consistent(accepted, rejected_policy="rollback")
    assert multiplier_contract_consistent(rolled_back, rejected_policy="rollback")
    assert not multiplier_contract_consistent(false_pass, rejected_policy="rollback")
    assert multiplier_contract_consistent(retained, rejected_policy="retain_full")


def test_rejected_public_position_cannot_bypass_multiplier_contract() -> None:
    rejected = {
        "line_search_rejected": True,
        "line_search_step": 0.0,
        "line_search_objective_before": 2.0,
        "line_search_objective_after": 2.0,
        "lagrangian_acceptance_policy": "retain_full_public_multiplier",
        "lagrangian_update_fraction": 1.0,
        "lagrangian_fraction_matches_observed": True,
        "lagrangian_transaction_relative_error": 0.0,
    }
    assert public_line_search_record_consistent(rejected)
    rejected["lagrangian_update_fraction"] = 0.0
    assert not public_line_search_record_consistent(rejected)


def _complete_sqp_record(*, rejected: bool = False) -> dict[str, object]:
    step = 0.0 if rejected else 0.25
    policy = (
        "rollback_multiplier_with_position"
        if rejected
        else "accepted_scaled_trial_multiplier"
    )
    merit_before = 2.0
    merit_slope = -4.0
    armijo_coefficient = 1.0e-4
    armijo_rhs = merit_before + armijo_coefficient * step * merit_slope
    return {
        "line_search_rejected": rejected,
        "line_search_step": step,
        "correction_global_scale": 1.0,
        "position_and_lagrangian_step_accepted_atomically": True,
        "lagrangian_acceptance_policy": policy,
        "lagrangian_update_fraction": step,
        "lagrangian_fraction_matches_observed": True,
        "lagrangian_transaction_relative_error": 0.0,
        "sqp_direction": {
            "enabled": True,
            "backend": "Torch_ROCm_MGPCG_Schur_active_set",
            "converged": True,
            "fallback_used": False,
            "active_set_iterations": 3,
            "active_constraints": 2,
            "maximum_auxiliary_true_relative_residual": 5.0e-6,
            "auxiliary_linear_solves": 2,
            "auxiliary_columns_computed": 2,
            "auxiliary_initial_linear_solves": 2,
            "auxiliary_refinement_linear_solves": 0,
            "auxiliary_pcg_iterations_total": 20,
            "auxiliary_zero_rhs_columns": 0,
            "auxiliary_final_active_columns": 2,
            "final_maximum_linearized_violation": 1.0e-6,
            "minimum_multiplier": 0.0,
            "active_equality_residual_maximum": 5.0e-7,
            "complementarity_maximum": 5.0e-7,
            "stationarity_relative": 5.0e-7,
            "schur_residual_relative": 5.0e-7,
            "schur_minimum_eigenvalue": 0.0,
            "coupled_linearized_residual_relative": 5.0e-7,
            "linearized_material_residual_l2": 5.0e-7,
            "direction_change_l2": 0.25,
            "kkt_passed": True,
            "configuration": {
                "primal_tolerance": 2.0e-5,
                "dual_tolerance": 1.0e-8,
                "kkt_relative_tolerance": 1.0e-6,
                "auxiliary_relative_residual_tolerance": 1.0e-5,
            },
            "nonlinear_safety_resolves": 1,
            "nonlinear_strain_cuts": 0,
            "nonlinear_determinant_cuts": 0,
            "full_direction_safety_feasible": True,
            "coupled_position_multiplier_transaction": True,
            "accepted_step": step,
            "accepted_multiplier_fraction": step,
            "multiplier_acceptance_policy": policy,
            "armijo_coefficient": armijo_coefficient,
            "armijo_merit_before": merit_before,
            "armijo_merit_after": merit_before if rejected else 1.5,
            "merit_slope": merit_slope,
            "armijo_rhs": armijo_rhs,
            "armijo_satisfied": not rejected,
            "rolled_back_atomically": True,
        },
    }


def test_scaled_sqp_multiplier_gate_checks_nested_and_observed_transaction() -> None:
    accepted = _complete_sqp_record()
    rejected = _complete_sqp_record(rejected=True)
    assert scaled_sqp_multiplier_contract_consistent(accepted)
    assert scaled_sqp_multiplier_contract_consistent(rejected)

    accepted["lagrangian_update_fraction"] = 1.0
    assert not scaled_sqp_multiplier_contract_consistent(accepted)
    accepted = _complete_sqp_record()
    accepted["lagrangian_acceptance_policy"] = "accepted_full_trial_multiplier"
    assert not scaled_sqp_multiplier_contract_consistent(accepted)
    accepted = _complete_sqp_record()
    del accepted["sqp_direction"]
    assert not scaled_sqp_multiplier_contract_consistent(accepted)


def test_sqp_armijo_gate_accepts_step_or_atomic_rollback_only() -> None:
    config = projector_configuration("orientation_safe_sqp_matrix_ua")
    accepted = _complete_sqp_record()
    rejected = _complete_sqp_record(rejected=True)
    assert sqp_armijo_and_atomic_receipt_consistent(accepted, config)
    assert sqp_armijo_and_atomic_receipt_consistent(rejected, config)

    accepted["sqp_direction"]["armijo_satisfied"] = False
    assert not sqp_armijo_and_atomic_receipt_consistent(accepted, config)
    accepted = _complete_sqp_record()
    accepted["sqp_direction"]["armijo_rhs"] = np.nan
    assert not sqp_armijo_and_atomic_receipt_consistent(accepted, config)
    rejected["sqp_direction"]["rolled_back_atomically"] = False
    assert not sqp_armijo_and_atomic_receipt_consistent(rejected, config)


def test_sqp_numerical_gate_is_fail_closed_on_missing_or_invalid_evidence() -> None:
    config = projector_configuration("orientation_safe_sqp_matrix_ua")
    complete = _complete_sqp_record()
    assert sqp_numerical_receipt_consistent(complete, config)
    assert sqp_receipt_contract_consistent(complete, config)

    missing_kkt = _complete_sqp_record()
    del missing_kkt["sqp_direction"]["stationarity_relative"]
    assert not sqp_numerical_receipt_consistent(missing_kkt, config)
    high_auxiliary_residual = _complete_sqp_record()
    high_auxiliary_residual["sqp_direction"][
        "maximum_auxiliary_true_relative_residual"
    ] = 2.0e-5
    assert not sqp_numerical_receipt_consistent(
        high_auxiliary_residual, config
    )
    missing_auxiliary_count = _complete_sqp_record()
    del missing_auxiliary_count["sqp_direction"][
        "auxiliary_refinement_linear_solves"
    ]
    assert not sqp_numerical_receipt_consistent(
        missing_auxiliary_count, config
    )
    inconsistent_auxiliary_count = _complete_sqp_record()
    inconsistent_auxiliary_count["sqp_direction"][
        "auxiliary_linear_solves"
    ] = 3
    assert not sqp_numerical_receipt_consistent(
        inconsistent_auxiliary_count, config
    )
    missing_coupling = _complete_sqp_record()
    del missing_coupling["sqp_direction"][
        "coupled_linearized_residual_relative"
    ]
    assert not sqp_numerical_receipt_consistent(missing_coupling, config)
    missing_nonlinear_receipt = _complete_sqp_record()
    del missing_nonlinear_receipt["sqp_direction"][
        "nonlinear_determinant_cuts"
    ]
    assert not sqp_numerical_receipt_consistent(
        missing_nonlinear_receipt, config
    )
    unsafe_full_direction = _complete_sqp_record()
    unsafe_full_direction["sqp_direction"][
        "full_direction_safety_feasible"
    ] = False
    assert not sqp_numerical_receipt_consistent(unsafe_full_direction, config)
    nonfinite_schur = _complete_sqp_record()
    nonfinite_schur["sqp_direction"]["schur_minimum_eigenvalue"] = np.nan
    assert not sqp_numerical_receipt_consistent(nonfinite_schur, config)


def test_projector_records_current_rap_and_true_residual() -> None:
    torch = pytest.importorskip("torch")
    scipy_spatial = pytest.importorskip("scipy.spatial")
    pytest.importorskip("pyamg")

    axis = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    rest = np.asarray(
        [(x, y, z) for x in axis for y in axis for z in axis],
        dtype=np.float32,
    )
    elements = np.asarray(scipy_spatial.Delaunay(rest).simplices, dtype=np.int32)
    elements = elements[
        np.abs(signed_six_volumes(rest, elements)) > 1.0e-10
    ]
    negative = signed_six_volumes(rest, elements) < 0.0
    elements[negative, 1], elements[negative, 2] = (
        elements[negative, 2].copy(),
        elements[negative, 1].copy(),
    )
    config = MGPBDTetConfig(
        dt_s=0.01,
        shear_modulus_pa=1.0e4,
        nonlinear_iterations=2,
        pcg_iterations=100,
        relative_residual=1.0e-5,
        amg_coarsest_size=4,
        amg_max_levels=4,
        amg_min_active_fraction=0.01,
        amg_hierarchy_mode="matrix_ua",
        symmetric_diagonal_equilibration=False,
        smoother_weight_mode="fine_spectral_radius",
        maximum_correction_m=None,
        line_search_acceptance_epsilon=0.0,
        line_search_minimum_step=1.0e-9,
        orientation_diagnostics_enabled=True,
    )
    rest_tensor = torch.as_tensor(rest, dtype=torch.float32)
    projector = VolumetricMGPBDProjector(
        rest_tensor,
        elements,
        vertex_masses=np.ones(len(rest)),
        config=config,
    )
    deformed = rest.copy()
    deformed[:, 1] *= 0.8
    projector.project(torch.as_tensor(deformed, dtype=torch.float32))
    records = projector.last_metrics["outer_iterations"]
    assert records
    assert projector.last_metrics["amg_rap_numeric_refreshes_last_project"] == len(
        records
    )
    for record in records:
        assert record["rap_refreshed_for_current_matrix"]
        assert np.isfinite(record["pcg_true_relative_residual"])
        assert record["level0_physical_operator_relative_error"] <= 5.0e-5
