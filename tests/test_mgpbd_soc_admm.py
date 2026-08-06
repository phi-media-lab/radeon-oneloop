import numpy as np
import pytest

from sim.genesis_so101.mgpbd_soc_admm import (
    SOCADMMConfig,
    SOCADMMConvergenceError,
    _dynamic_kkt_pcg_target,
    _normal_cone_metrics,
    apply_deformation_jacobian,
    apply_deformation_jacobian_transpose,
    closest_proper_rotations,
    deformation_gradients,
    deterministic_vertex_sum,
    project_frobenius_balls,
    solve_soc_admm_direction,
    vertex_incidence_slots,
)


def _two_tet_fixture(*, dtype=None):
    """Return two irregular, face-sharing tetrahedra in positive orientation."""

    torch = pytest.importorskip("torch")
    from sim.genesis_so101.mgpbd_tet import tetrahedral_rest_data

    dtype = dtype or torch.float64
    rest_np = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    elements_np = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int64)
    rest_inverse_np, _rest_volumes = tetrahedral_rest_data(
        rest_np, elements_np
    )
    current_np = rest_np.copy()
    current_np[2] += np.asarray((0.03, -0.06, 0.02))
    current_np[3] += np.asarray((-0.02, 0.01, -0.08))
    current_np[4] += np.asarray((0.08, -0.07, 0.04))
    generator = torch.Generator().manual_seed(20260806)
    return {
        "rest_np": rest_np,
        "current": torch.as_tensor(current_np, dtype=dtype),
        "elements": torch.as_tensor(elements_np, dtype=torch.long),
        "rest_inverse": torch.as_tensor(rest_inverse_np, dtype=dtype),
        "masses": torch.as_tensor((1.0, 1.2, 0.8, 1.1, 0.9), dtype=dtype),
        "material_gradients": torch.randn(
            (2, 4, 3), generator=generator, dtype=dtype
        ),
        "q": torch.as_tensor((-0.7, 0.3), dtype=dtype),
        "alpha": torch.as_tensor((0.2, 0.3), dtype=dtype),
    }


def _torch_fixture(*, dtype=None):
    torch = pytest.importorskip("torch")
    dtype = dtype or torch.float64
    current = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=dtype,
    )
    elements = torch.tensor(((0, 1, 2, 3),), dtype=torch.long)
    rest_inverse = torch.eye(3, dtype=dtype)[None]
    masses = torch.ones(4, dtype=dtype)
    material_gradients = torch.zeros((1, 4, 3), dtype=dtype)
    material_gradients[0, 0, 0] = -1.0
    material_gradients[0, 1, 0] = 1.0
    return {
        "current": current,
        "elements": elements,
        "rest_inverse": rest_inverse,
        "masses": masses,
        "material_gradients": material_gradients,
    }


def test_soc_admm_config_leaves_strict_determinant_margin() -> None:
    config = SOCADMMConfig()
    assert config.proof_radius == pytest.approx(0.99)
    assert config.beta == pytest.approx(1.0e-3)
    assert not config.scale_beta_by_operator_diagonal
    assert config.beta_update_interval == 25
    assert config.maximum_admm_iterations == 2_000
    assert config.admm_primal_tolerance == pytest.approx(2.0e-4)
    with pytest.raises(ValueError, match="proof margin"):
        SOCADMMConfig(work_radius=0.99).validate()


def test_deformation_jacobian_and_transpose_are_adjoint() -> None:
    torch = pytest.importorskip("torch")
    fixture = _torch_fixture()
    generator = torch.Generator().manual_seed(20260806)
    direction = torch.randn((4, 3), generator=generator, dtype=torch.float64)
    matrices = torch.randn((1, 3, 3), generator=generator, dtype=torch.float64)
    applied = apply_deformation_jacobian(
        direction, fixture["rest_inverse"], fixture["elements"]
    )
    transposed = apply_deformation_jacobian_transpose(
        matrices,
        fixture["rest_inverse"],
        fixture["elements"],
        len(direction),
    )
    torch.testing.assert_close(
        torch.sum(applied * matrices),
        torch.sum(direction * transposed),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_face_sharing_multi_tet_jacobian_and_transpose_are_adjoint() -> None:
    """Exercise gather/scatter overlap, which a one-tet test cannot cover."""

    torch = pytest.importorskip("torch")
    fixture = _two_tet_fixture()
    generator = torch.Generator().manual_seed(20260807)
    direction = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    matrices = torch.randn((2, 3, 3), generator=generator, dtype=torch.float64)
    applied = apply_deformation_jacobian(
        direction, fixture["rest_inverse"], fixture["elements"]
    )
    transposed = apply_deformation_jacobian_transpose(
        matrices,
        fixture["rest_inverse"],
        fixture["elements"],
        len(direction),
    )
    torch.testing.assert_close(
        torch.sum(applied * matrices),
        torch.sum(direction * transposed),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_fixed_order_incidence_reduction_matches_scatter_oracle() -> None:
    torch = pytest.importorskip("torch")
    fixture = _two_tet_fixture()
    generator = torch.Generator().manual_seed(20260808)
    matrices = torch.randn((2, 3, 3), generator=generator, dtype=torch.float64)
    slots = vertex_incidence_slots(fixture["elements"], 5)
    scattered = apply_deformation_jacobian_transpose(
        matrices,
        fixture["rest_inverse"],
        fixture["elements"],
        5,
    )
    gathered = apply_deformation_jacobian_transpose(
        matrices,
        fixture["rest_inverse"],
        fixture["elements"],
        5,
        incidence_slots=slots,
    )
    torch.testing.assert_close(gathered, scattered, rtol=1.0e-13, atol=1.0e-13)
    assert torch.equal(
        gathered,
        apply_deformation_jacobian_transpose(
            matrices,
            fixture["rest_inverse"],
            fixture["elements"],
            5,
            incidence_slots=slots,
        ),
    )

    local_values = torch.randn(
        (2, 4, 3), generator=generator, dtype=torch.float64
    )
    scatter_sum = torch.zeros((5, 3), dtype=torch.float64)
    for local_index in range(4):
        scatter_sum.index_add_(
            0,
            fixture["elements"][:, local_index],
            local_values[:, local_index],
        )
    torch.testing.assert_close(
        deterministic_vertex_sum(local_values, slots),
        scatter_sum,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_fixed_rotation_ball_majorizes_true_arap_and_proves_orientation() -> None:
    """Check the geometric implication used by every SOC block."""

    torch = pytest.importorskip("torch")
    rest = _torch_fixture()["current"]
    elements = torch.tensor(((0, 1, 2, 3),), dtype=torch.long)
    rest_inverse = torch.eye(3, dtype=torch.float64)[None]
    fixed_rotation, _distance = closest_proper_rotations(
        deformation_gradients(rest, rest_inverse, elements)
    )

    # Include both a substantially rotated candidate and a candidate close to
    # the determinant boundary.  Distance to SO(3) is the minimum over proper
    # rotations, so the frozen-rotation distance must be an upper bound.
    angle = torch.tensor(np.pi / 3.0, dtype=torch.float64)
    rotated = rest.clone()
    rotated[1] = torch.stack((torch.cos(angle), torch.sin(angle), angle * 0.0))
    rotated[2] = torch.stack((-torch.sin(angle), torch.cos(angle), angle * 0.0))
    near_boundary = rest.clone()
    near_boundary[1, 0] = 0.011
    for candidate in (rotated, near_boundary):
        deformation = deformation_gradients(
            candidate, rest_inverse, elements
        )
        _closest, true_distance = closest_proper_rotations(deformation)
        frozen_distance = torch.linalg.vector_norm(
            deformation - fixed_rotation, dim=(1, 2)
        )
        assert float(true_distance[0]) <= float(frozen_distance[0]) + 1.0e-13

    boundary_deformation = deformation_gradients(
        near_boundary, rest_inverse, elements
    )
    frozen_norm = float(
        torch.linalg.vector_norm(
            boundary_deformation - fixed_rotation, dim=(1, 2)
        )[0]
    )
    determinant = float(torch.linalg.det(boundary_deformation)[0])
    assert frozen_norm == pytest.approx(0.989, abs=1.0e-13)
    assert determinant > 0.0
    assert determinant >= (1.0 - frozen_norm) ** 3


def test_normal_cone_gate_uses_physical_dual_not_beta_scaled_dual() -> None:
    """Adaptive beta must not silently rescale the complementarity gate."""

    torch = pytest.importorskip("torch")
    radius = 0.989
    z = torch.zeros((2, 3, 3), dtype=torch.float64)
    z[0, 0, 0] = radius
    physical_dual = torch.zeros_like(z)
    physical_dual[0, 0, 0] = 0.2
    physical_dual[0, 0, 1] = 1.0e-3
    physical_dual[1, 0, 0] = 2.0e-3

    small_beta = _normal_cone_metrics(
        z,
        physical_dual,
        radius,
        scaled_dual=physical_dual / 1.0e-4,
    )
    large_beta = _normal_cone_metrics(
        z,
        physical_dual,
        radius,
        scaled_dual=physical_dual / 1.0e-1,
    )
    assert small_beta["dual_convention"] == (
        "physical_y_equals_beta_times_scaled_dual"
    )
    assert small_beta["gate_residual"] == pytest.approx(
        large_beta["gate_residual"], rel=0.0, abs=0.0
    )
    assert small_beta["gate_residual"] == pytest.approx(
        np.sqrt(5.0e-6), rel=1.0e-12
    )
    # The scaled-ADMM fixed-point diagnostic is intentionally allowed to vary;
    # it is no longer used as the physical normal-cone gate.
    assert small_beta["maximum_projection_fixed_point_residual"] != pytest.approx(
        large_beta["maximum_projection_fixed_point_residual"]
    )


def test_dynamic_kkt_pcg_target_is_current_budget_not_iteration_history() -> None:
    config = SOCADMMConfig()
    common = {
        "stationarity_score": 2.0,
        "primal_score": 1.0,
        "dual_score": 1.0,
        "normal_residual": 0.0,
        "proof_maximum": config.work_radius,
        "config": config,
    }
    target = _dynamic_kkt_pcg_target(
        **common,
        kkt_target_l2=10.0,
        dual_vector_l2=2.0,
    )
    assert target == pytest.approx(6.0)
    # Re-evaluating the same state cannot geometrically halve a historical
    # target, while a larger current budget is allowed to relax it.
    assert _dynamic_kkt_pcg_target(
        **common,
        kkt_target_l2=10.0,
        dual_vector_l2=2.0,
    ) == pytest.approx(target)
    assert _dynamic_kkt_pcg_target(
        **common,
        kkt_target_l2=12.0,
        dual_vector_l2=2.0,
    ) == pytest.approx(7.5)


def test_dynamic_kkt_pcg_target_only_uses_force_space_budget() -> None:
    config = SOCADMMConfig()
    common = {
        "primal_score": 1.0,
        "dual_score": 1.0,
        "normal_residual": 0.0,
        "proof_maximum": config.work_radius,
        "kkt_target_l2": 1.0,
        "config": config,
    }
    assert _dynamic_kkt_pcg_target(
        **common,
        stationarity_score=1.0,
        dual_vector_l2=0.0,
    ) is None
    assert _dynamic_kkt_pcg_target(
        **common,
        stationarity_score=1.0,
        dual_vector_l2=0.0,
        previous_target_l2=0.25,
        confirmation_pass_active=True,
    ) == pytest.approx(0.25)
    assert _dynamic_kkt_pcg_target(
        **common,
        stationarity_score=2.0,
        dual_vector_l2=1.0,
    ) is None


def test_dynamic_kkt_pcg_target_does_not_toggle_off_after_polish_starts() -> None:
    """A transiently spent dual budget must retain the prior force target."""

    config = SOCADMMConfig()
    common = {
        "stationarity_score": 2.0,
        "primal_score": 1.0,
        "dual_score": 1.0,
        "normal_residual": 0.0,
        "proof_maximum": config.work_radius,
        "config": config,
        "polish_active": True,
    }
    retained = _dynamic_kkt_pcg_target(
        **common,
        kkt_target_l2=1.0,
        dual_vector_l2=1.0,
        previous_target_l2=0.25,
    )
    assert retained == pytest.approx(0.25)
    # A later positive current budget can safely relax the target; this is
    # persistence, not the historical geometric-halving failure.
    assert _dynamic_kkt_pcg_target(
        **common,
        kkt_target_l2=2.0,
        dual_vector_l2=1.0,
        previous_target_l2=retained,
    ) == pytest.approx(0.75)


def test_dynamic_kkt_confirmation_never_relaxes_the_passing_target() -> None:
    config = SOCADMMConfig()
    target = _dynamic_kkt_pcg_target(
        stationarity_score=2.0,
        primal_score=1.0,
        dual_score=1.0,
        normal_residual=0.0,
        proof_maximum=config.work_radius,
        kkt_target_l2=2.0,
        dual_vector_l2=1.0,
        config=config,
        previous_target_l2=0.25,
        polish_active=True,
        confirmation_pass_active=True,
    )
    assert target == pytest.approx(0.25)


def test_frobenius_ball_projection_is_block_local() -> None:
    torch = pytest.importorskip("torch")
    values = torch.zeros((2, 3, 3), dtype=torch.float64)
    values[0, 0, 0] = 0.25
    values[1, 0, 0] = 3.0
    values[1, 1, 1] = 4.0
    projected = project_frobenius_balls(values, 1.0)
    norms = torch.linalg.vector_norm(projected, dim=(1, 2))
    torch.testing.assert_close(
        norms, torch.tensor((0.25, 1.0), dtype=torch.float64)
    )
    torch.testing.assert_close(projected[0], values[0])
    torch.testing.assert_close(projected[1], values[1] / 5.0)


def test_pcg_does_not_recompute_a_true_residual_after_it_passes() -> None:
    """Keep the exact residual that triggered the strict convergence gate."""

    torch = pytest.importorskip("torch")
    from sim.genesis_so101.mgpbd_soc_admm import _matrix_free_pcg

    calls = 0

    def identity(values):
        nonlocal calls
        calls += 1
        if calls > 3:
            raise AssertionError("passed true residual was recomputed")
        return values

    solution, receipt = _matrix_free_pcg(
        identity,
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        SOCADMMConfig(
            adaptive_beta=False,
            pcg_maximum_iterations=10,
            pcg_relative_tolerance=1.0e-12,
            pcg_absolute_tolerance=1.0e-14,
        ),
    )
    torch.testing.assert_close(solution, torch.ones_like(solution))
    assert receipt["converged"]
    assert receipt["true_residual_l2"] == 0.0
    assert calls == 3


def test_pcg_target_override_enforces_the_tighter_absolute_budget() -> None:
    """The KKT polish budget must dominate a loose RHS-relative target."""

    torch = pytest.importorskip("torch")
    from sim.genesis_so101.mgpbd_soc_admm import _matrix_free_pcg

    solution, receipt = _matrix_free_pcg(
        lambda values: values,
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        SOCADMMConfig(
            adaptive_beta=False,
            pcg_maximum_iterations=10,
            pcg_relative_tolerance=1.0,
            pcg_absolute_tolerance=1.0e-12,
        ),
        target_l2_override=0.1,
    )
    torch.testing.assert_close(solution, torch.ones_like(solution))
    assert receipt["converged"]
    assert receipt["default_target_l2"] == pytest.approx(1.0)
    assert receipt["target_l2_override"] == pytest.approx(0.1)
    assert receipt["target_l2"] == pytest.approx(0.1)
    assert receipt["iterations"] == 1


def test_zero_material_residual_returns_zero_safe_direction() -> None:
    torch = pytest.importorskip("torch")
    fixture = _torch_fixture()
    config = SOCADMMConfig(
        scale_beta_by_operator_diagonal=False,
        adaptive_beta=False,
    )
    result = solve_soc_admm_direction(
        **fixture,
        q=torch.zeros(1, dtype=torch.float64),
        alpha=torch.ones(1, dtype=torch.float64),
        config=config,
    )
    torch.testing.assert_close(result.direction, torch.zeros_like(result.direction))
    torch.testing.assert_close(
        result.delta_lambda, torch.zeros_like(result.delta_lambda)
    )
    assert result.metrics["passed"]
    assert (
        result.metrics["admm_iterations"]
        == config.required_consecutive_gate_passes
    )
    assert result.metrics["effective_beta_history"] == [
        1.0e-3
    ] * config.required_consecutive_gate_passes


def test_active_soc_direction_is_safe_and_audits_adaptive_beta() -> None:
    torch = pytest.importorskip("torch")
    fixture = _torch_fixture()
    config = SOCADMMConfig(
        beta=1.0e-3,
        beta_maximum=1.0e5,
        scale_beta_by_operator_diagonal=False,
        adaptive_beta=True,
        beta_update_interval=2,
        beta_balance_ratio=2.0,
        beta_update_factor=2.0,
        maximum_admm_iterations=2_000,
        admm_primal_tolerance=1.0e-7,
        admm_dual_relative_tolerance=1.0e-7,
        stationarity_relative_tolerance=1.0e-7,
        normal_cone_tolerance=1.0e-7,
        pcg_relative_tolerance=1.0e-10,
        pcg_absolute_tolerance=1.0e-12,
    )
    result = solve_soc_admm_direction(
        **fixture,
        q=torch.tensor((2.0,), dtype=torch.float64),
        alpha=torch.tensor((1.0e-3,), dtype=torch.float64),
        config=config,
    )
    candidate_deformation = deformation_gradients(
        fixture["current"] + result.direction,
        fixture["rest_inverse"],
        fixture["elements"],
    )
    assert result.metrics["passed"]
    assert result.metrics["fixed_rotation_candidate_norm_maximum"] == pytest.approx(
        config.work_radius, abs=3.0e-7
    )
    assert result.metrics["true_arap_maximum"] <= 1.0
    assert result.metrics["minimum_signed_volume_ratio"] >= 1.0e-6
    assert float(torch.linalg.det(candidate_deformation)[0]) >= 1.0e-6
    assert len(result.metrics["effective_beta_history"]) == result.metrics[
        "admm_iterations"
    ]
    assert result.metrics["adaptive_beta_update_count"] == len(
        result.metrics["adaptive_beta_updates"]
    )
    assert result.metrics["adaptive_beta_update_count"] > 0
    for update in result.metrics["adaptive_beta_updates"]:
        history_index = int(update["after_admm_iteration"]) - 1
        assert update["dual_gate_score"] == pytest.approx(
            result.metrics["admm_dual_residual_relative_history"][history_index]
            / config.admm_dual_relative_tolerance
        )
        assert update["stationarity_gate_score"] == pytest.approx(
            result.metrics["stationarity_relative_history"][history_index]
            / config.stationarity_relative_tolerance
        )
        old_beta = float(update["old_effective_beta"])
        new_beta = float(update["new_effective_beta"])
        primal_score = float(update["primal_gate_score"])
        dual_score = float(update["dual_gate_score"])
        opposition_score = float(update["beta_balance_opposition_score"])
        assert opposition_score == pytest.approx(
            max(dual_score, float(update["stationarity_gate_score"]))
        )
        if update["reason"] == "primal_gate_dominates":
            assert primal_score > 1.0
            assert (
                primal_score
                > config.beta_balance_ratio * opposition_score
            )
            assert new_beta == pytest.approx(
                min(old_beta * config.beta_update_factor, config.beta_maximum)
            )
        else:
            assert update["reason"] == "dual_or_kkt_guard_dominates"
            assert opposition_score > 1.0
            assert (
                opposition_score
                > config.beta_balance_ratio * primal_score
            )
            assert new_beta == pytest.approx(
                max(old_beta / config.beta_update_factor, config.beta_minimum)
            )


def test_float32_precision_continuation_is_reaudited_after_one_final_cast(
    monkeypatch,
) -> None:
    """Exercise the real full-state continuation branch, not only its schema."""

    torch = pytest.importorskip("torch")
    import sim.genesis_so101.mgpbd_soc_admm as soc_module

    fixture = _torch_fixture(dtype=torch.float32)
    original_pcg = soc_module._matrix_free_pcg
    forced_floor = {"triggered": False}

    def force_one_finite_float32_floor(
        operator,
        rhs,
        diagonal,
        initial,
        config,
        *,
        target_l2_override=None,
    ):
        solution, receipt = original_pcg(
            operator,
            rhs,
            diagonal,
            initial,
            config,
            target_l2_override=target_l2_override,
        )
        if (
            not forced_floor["triggered"]
            and rhs.dtype == torch.float32
            and target_l2_override is not None
        ):
            forced_floor["triggered"] = True
            target = float(receipt["target_l2"])
            receipt = {
                **receipt,
                "converged": False,
                "breakdown": None,
                "true_residual_l2": 2.0 * target,
                "true_residual_to_target": 2.0,
            }
        return solution, receipt

    monkeypatch.setattr(
        soc_module, "_matrix_free_pcg", force_one_finite_float32_floor
    )
    result = solve_soc_admm_direction(
        **fixture,
        q=torch.tensor((2.0,), dtype=torch.float32),
        alpha=torch.tensor((1.0e-3,), dtype=torch.float32),
        config=SOCADMMConfig(
            beta=1.0e-3,
            beta_maximum=1.0e2,
            adaptive_beta=True,
            beta_update_interval=2,
            beta_balance_ratio=2.0,
            maximum_admm_iterations=2_000,
            required_consecutive_gate_passes=2,
            stationarity_relative_tolerance=1.0e-6,
            pcg_maximum_iterations=500,
            pcg_relative_tolerance=1.0e-5,
            pcg_absolute_tolerance=1.0e-10,
        ),
    )
    assert forced_floor["triggered"]
    assert result.direction.dtype == torch.float32
    assert result.delta_lambda.dtype == torch.float32
    assert result.metrics["precision_continuation_active"]
    assert result.metrics["precision_continuation_events"]
    assert result.metrics["stationarity_gate_tolerance_history"][-1] == (
        pytest.approx(
            0.99
            * result.metrics["configuration"][
                "stationarity_relative_tolerance"
            ]
        )
    )
    assert result.metrics["checks"]["accepted_dtype_reaudit_satisfied"]
    assert result.metrics["accepted_dtype_reaudit"]["passed"]


def test_face_sharing_multi_tet_matches_independent_sparse_direct_oracle() -> None:
    """Compare the matrix-free direction with independently assembled SciPy K/J."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from sim.genesis_so101.mgpbd_soc_admm_oracle import solve_soc_admm_numpy

    fixture = _two_tet_fixture()
    config = SOCADMMConfig(
        beta=1.0e-2,
        scale_beta_by_operator_diagonal=False,
        adaptive_beta=False,
        maximum_admm_iterations=10_000,
        required_consecutive_gate_passes=3,
        admm_primal_tolerance=1.0e-8,
        admm_dual_relative_tolerance=1.0e-8,
        stationarity_relative_tolerance=1.0e-8,
        normal_cone_tolerance=1.0e-8,
        coupled_material_relative_tolerance=1.0e-10,
        pcg_maximum_iterations=500,
        pcg_relative_tolerance=1.0e-12,
        pcg_absolute_tolerance=1.0e-14,
        pcg_residual_replacement_interval=10,
    )
    torch_result = solve_soc_admm_direction(
        current=fixture["current"],
        elements=fixture["elements"],
        rest_inverse=fixture["rest_inverse"],
        masses=fixture["masses"],
        material_gradients=fixture["material_gradients"],
        q=fixture["q"],
        alpha=fixture["alpha"],
        config=config,
    )

    rest_np = fixture["rest_np"]
    elements_np = fixture["elements"].cpu().numpy()
    tets = rest_np[elements_np]
    rest_signed_six = np.linalg.det(
        np.stack(
            (
                tets[:, 1] - tets[:, 0],
                tets[:, 2] - tets[:, 0],
                tets[:, 3] - tets[:, 0],
            ),
            axis=2,
        )
    )
    oracle = solve_soc_admm_numpy(
        current=fixture["current"].cpu().numpy(),
        elements=elements_np,
        rest_inverse=fixture["rest_inverse"].cpu().numpy(),
        rest_signed_six=rest_signed_six,
        material_constraints=fixture["q"].cpu().numpy(),
        material_gradients=fixture["material_gradients"].cpu().numpy(),
        material_compliance=fixture["alpha"].cpu().numpy(),
        vertex_masses=fixture["masses"].cpu().numpy(),
        work_radius=config.work_radius,
        safe_radius=config.proof_radius,
        penalty=config.beta,
        maximum_iterations=config.maximum_admm_iterations,
        primal_tolerance=config.admm_primal_tolerance,
        dual_tolerance=config.admm_dual_relative_tolerance,
        stationarity_tolerance=config.stationarity_relative_tolerance,
        penalty_update_interval=config.beta_update_interval,
        penalty_balance_ratio=config.beta_balance_ratio,
        penalty_scale=config.beta_update_factor,
        progress_interval=config.maximum_admm_iterations + 1,
    )
    oracle_direction = torch.as_tensor(
        oracle["direction"], dtype=torch_result.direction.dtype
    )
    torch.testing.assert_close(
        torch_result.direction,
        oracle_direction,
        rtol=5.0e-5,
        atol=5.0e-6,
    )
    torch.testing.assert_close(
        torch_result.delta_lambda,
        torch.as_tensor(
            oracle["delta_lambda"], dtype=torch_result.delta_lambda.dtype
        ),
        rtol=5.0e-5,
        atol=5.0e-6,
    )

    def objective(direction):
        local_direction = direction[fixture["elements"]]
        material = fixture["q"] + torch.sum(
            fixture["material_gradients"] * local_direction, dim=(1, 2)
        )
        return 0.5 * torch.sum(
            fixture["masses"][:, None] * direction * direction
        ) + 0.5 * torch.sum(material * material / fixture["alpha"])

    torch.testing.assert_close(
        objective(torch_result.direction),
        objective(oracle_direction),
        rtol=1.0e-7,
        atol=1.0e-9,
    )


def test_active_soc_iteration_limit_fails_closed_with_receipt() -> None:
    torch = pytest.importorskip("torch")
    fixture = _torch_fixture()
    with pytest.raises(SOCADMMConvergenceError) as caught:
        solve_soc_admm_direction(
            **fixture,
            q=torch.tensor((2.0,), dtype=torch.float64),
            alpha=torch.tensor((1.0e-3,), dtype=torch.float64),
            config=SOCADMMConfig(
                scale_beta_by_operator_diagonal=False,
                adaptive_beta=False,
                maximum_admm_iterations=1,
            ),
        )
    assert caught.value.receipt["failure"] == "admm_iteration_limit"
    assert not caught.value.receipt["converged"]
    assert caught.value.receipt["effective_beta_history"] == [1.0e-3]


def test_current_state_outside_work_ball_fails_before_solve() -> None:
    torch = pytest.importorskip("torch")
    fixture = _torch_fixture()
    fixture["current"] = fixture["current"].clone()
    fixture["current"][1, 0] = 3.0
    with pytest.raises(SOCADMMConvergenceError) as caught:
        solve_soc_admm_direction(
            **fixture,
            q=torch.zeros(1, dtype=torch.float64),
            alpha=torch.ones(1, dtype=torch.float64),
        )
    assert caught.value.receipt["failure"] == "current_state_outside_work_ball"
