import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from sim.genesis_so101.mgpbd_tet import (
    MGPBDTetConfig,
    VolumetricMGPBDProjector,
    armijo_merit_rejected,
    arap_constraints_and_gradients_numpy,
    boundary_faces,
    distal_finger_contact_vertex_indices,
    greedy_unsmoothed_aggregation,
    lagrangian_acceptance_policy,
    line_search_objective_rejected,
    strain_trust_filter_rejected,
    load_precomputed_tet_mesh,
    tetrahedral_rest_data,
    triangle_contact_samples,
    unique_tet_edges,
)


REST = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
ELEMENT = np.asarray(((0, 1, 2, 3),), dtype=np.int32)


def _write_dense_fixture(tmp_path, *, contact_radius_m: float = 0.00035):
    path = tmp_path / "volume.npz"
    manifest_path = tmp_path / "volume.json"
    np.savez_compressed(
        path,
        nodes=REST.astype(np.float32),
        elements=ELEMENT,
        boundary_faces=boundary_faces(ELEMENT),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "radeon_oneloop.mgpbd_dense_volume.v1",
                "quality_gate": {"passed": True},
                "volume": {
                    "sha256": digest,
                    "nodes": 4,
                    "tetrahedra": 1,
                },
                "runtime": {
                    "contact_radius_m": contact_radius_m,
                    "visual_binding": "tetrahedron_barycentric_embedding",
                },
            }
        ),
        encoding="utf-8",
    )
    return path, manifest_path


def test_load_precomputed_tet_mesh_verifies_archive(tmp_path) -> None:
    path, manifest = _write_dense_fixture(tmp_path)
    nodes, elements, diagnostics = load_precomputed_tet_mesh(
        path, manifest, expected_contact_radius_m=0.00035
    )
    np.testing.assert_allclose(nodes, REST)
    np.testing.assert_array_equal(elements, ELEMENT)
    assert diagnostics["physics_vertices"] == 4
    assert diagnostics["physics_tetrahedra"] == 1
    assert diagnostics["boundary_faces"] == 4


def test_load_precomputed_tet_mesh_rejects_contact_radius_mismatch(tmp_path) -> None:
    path, manifest = _write_dense_fixture(tmp_path)
    with pytest.raises(ValueError, match="contact radius"):
        load_precomputed_tet_mesh(
            path, manifest, expected_contact_radius_m=0.002
        )


def test_unique_tet_edges_welds_shared_edge_constraints() -> None:
    elements = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int32)
    edges = unique_tet_edges(elements)
    assert edges.shape == (9, 2)
    assert len({tuple(edge) for edge in edges.tolist()}) == 9
    assert (edges[:, 0] < edges[:, 1]).all()
    assert (np.asarray((1, 2)) == edges).all(axis=1).any()


def test_ua_aggregation_uses_weak_structural_neighbours() -> None:
    sparse = pytest.importorskip("scipy.sparse")

    matrix = sparse.diags(
        (
            np.full(7, 1.0e-6),
            np.ones(8),
            np.full(7, 1.0e-6),
        ),
        offsets=(-1, 0, 1),
        format="csr",
    )
    aggregate = greedy_unsmoothed_aggregation(matrix, threshold=0.1)
    assert len(aggregate) == 8
    assert int(aggregate.max()) + 1 < 8
    assert aggregate[0] == aggregate[1]


def test_mgpbd_config_alpha_uses_xpbd_timestep_scaling() -> None:
    config = MGPBDTetConfig(dt_s=0.01, shear_modulus_pa=2.0e4)
    config.validate()
    alpha = config.alpha_tilde(np.asarray((0.25, 0.50)))
    assert alpha.tolist() == pytest.approx((2.0, 1.0))


def test_mgpbd_config_rejects_unstable_relaxation() -> None:
    with pytest.raises(ValueError, match="relaxation"):
        MGPBDTetConfig(relaxation=1.1).validate()
    with pytest.raises(ValueError, match="minimum step"):
        MGPBDTetConfig(
            relaxation=0.1, line_search_minimum_step=0.1
        ).validate()


def test_sqp_auxiliary_refinement_configuration_is_fail_closed() -> None:
    assert MGPBDTetConfig().sqp_auxiliary_relative_residual == pytest.approx(
        1.0e-6
    )
    config = MGPBDTetConfig(
        sqp_auxiliary_relative_residual=1.0e-6,
        sqp_maximum_auxiliary_refinements=3,
    )
    config.validate()
    assert config.sqp_auxiliary_relative_residual == pytest.approx(1.0e-6)
    with pytest.raises(ValueError, match="KKT tolerances"):
        MGPBDTetConfig(sqp_auxiliary_relative_residual=0.0).validate()
    with pytest.raises(ValueError, match="active-set/cutting-plane"):
        MGPBDTetConfig(sqp_maximum_auxiliary_refinements=-1).validate()


def test_soc_admm_direction_configuration_is_explicit_and_guarded() -> None:
    default = MGPBDTetConfig()
    assert not default.soc_admm_direction_enabled
    inner = default.soc_admm_config()
    assert inner.beta == pytest.approx(default.soc_admm_beta)
    assert inner.kkt_polish_beta_maximum is None
    assert inner.work_radius == pytest.approx(default.soc_admm_work_radius)
    assert inner.accepted_dtype_stationarity_safety_factor == pytest.approx(
        default.soc_admm_accepted_dtype_stationarity_safety_factor
    )
    assert inner.maximum_admm_iterations == default.soc_admm_maximum_iterations
    assert inner.pcg_maximum_iterations == default.soc_admm_pcg_maximum_iterations

    with pytest.raises(ValueError, match="mutually exclusive"):
        MGPBDTetConfig(
            sqp_direction_enabled=True,
            soc_admm_direction_enabled=True,
        ).validate()
    with pytest.raises(ValueError, match="SOC-ADMM direction requires"):
        MGPBDTetConfig(soc_admm_direction_enabled=True).validate()
    guarded = _soc_enabled_projector_config()
    guarded.validate()
    with pytest.raises(ValueError, match="outer accepted-state strain gate"):
        replace(guarded, strain_trust_filter_maximum=1.0).validate()
    with pytest.raises(ValueError, match="proof margin"):
        MGPBDTetConfig(soc_admm_work_radius=0.999).validate()
    with pytest.raises(ValueError, match="KKT-polish beta cap"):
        MGPBDTetConfig(
            soc_admm_beta_minimum=1.0e-4,
            soc_admm_kkt_polish_beta_maximum=1.0e-5,
        ).validate()
    with pytest.raises(ValueError, match="stationarity safety factor"):
        replace(
            _soc_enabled_projector_config(),
            soc_admm_accepted_dtype_stationarity_safety_factor=1.01,
        ).validate()


def test_local_support_mass_gram_matches_dense_vertex_oracle() -> None:
    torch = pytest.importorskip("torch")

    projector = object.__new__(VolumetricMGPBDProjector)
    projector.device = torch.device("cpu")
    projector.elements = torch.as_tensor(
        (
            (0, 1, 2, 3),
            (0, 2, 1, 4),
            (0, 1, 5, 6),
            (0, 7, 8, 9),
            (10, 11, 12, 13),
        ),
        dtype=torch.long,
    )
    projector.inverse_mass = torch.linspace(
        0.25, 2.0, 14, dtype=torch.float32
    )
    row_tets = torch.as_tensor((0, 0, 1, 2, 3, 4, 1), dtype=torch.long)
    generator = torch.Generator().manual_seed(20_260_806)
    rows = torch.randn((len(row_tets), 4, 3), generator=generator)

    actual = projector._local_rows_mass_gram(rows, row_tets)

    dense64 = torch.zeros((len(row_tets), 14, 3), dtype=torch.float64)
    batch = torch.arange(len(row_tets))
    vertices = projector.elements[row_tets]
    for local_index in range(4):
        dense64[batch, vertices[:, local_index]] += rows[
            :, local_index
        ].to(torch.float64)
    weighted_dense64 = dense64 * projector.inverse_mass.to(torch.float64)[
        None, :, None
    ]
    expected = dense64.flatten(1) @ weighted_dense64.flatten(1).transpose(0, 1)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(actual, actual.transpose(0, 1), rtol=0.0, atol=1.0e-12)
    empty = projector._local_rows_mass_gram(
        rows[:0], torch.empty(0, dtype=torch.long)
    )
    assert empty.shape == (0, 0)
    assert empty.dtype == torch.float64


def test_public_line_search_requires_strict_objective_decrease() -> None:
    assert line_search_objective_rejected(1.0, 1.0, 0.0)
    assert line_search_objective_rejected(1.0 + 1.0e-8, 1.0, 0.0)
    assert not line_search_objective_rejected(1.0 - 1.0e-8, 1.0, 0.0)
    assert not line_search_objective_rejected(1.0 + 1.0e-8, 1.0, 1.0e-7)
    assert line_search_objective_rejected(np.nan, 1.0, 0.0)
    assert line_search_objective_rejected(1.0, np.inf, 0.0)


def test_coupled_sqp_armijo_uses_squared_dual_merit() -> None:
    assert not armijo_merit_rejected(1.0, 2.0, 0.5, -4.0, 1.0e-4)
    assert armijo_merit_rejected(2.0, 2.0, 0.5, -4.0, 1.0e-4)
    assert armijo_merit_rejected(np.nan, 2.0, 0.5, -4.0, 1.0e-4)
    assert armijo_merit_rejected(1.0, 2.0, 0.5, 0.0, 1.0e-4)
    assert not armijo_merit_rejected(0.0, 0.0, 1.0, 0.0, 1.0e-4)


def test_multiplier_acceptance_policies_are_explicit() -> None:
    assert lagrangian_acceptance_policy(
        rejected=False,
        rejected_policy="rollback",
        line_search_scale_lagrangian=False,
    ) == ("accepted_full_trial_multiplier", 1.0)
    assert lagrangian_acceptance_policy(
        rejected=True,
        rejected_policy="retain_full",
        line_search_scale_lagrangian=False,
    ) == ("retain_full_public_multiplier", 1.0)
    assert lagrangian_acceptance_policy(
        rejected=True,
        rejected_policy="rollback",
        line_search_scale_lagrangian=False,
    ) == ("rollback_multiplier_with_position", 0.0)
    assert lagrangian_acceptance_policy(
        rejected=False,
        rejected_policy="rollback",
        line_search_scale_lagrangian=True,
        accepted_step=0.25,
        correction_global_scale=0.5,
    ) == ("accepted_scaled_trial_multiplier", 0.125)


def test_strain_trust_filter_is_absolute_and_fail_closed() -> None:
    assert not strain_trust_filter_rejected(0.99, enabled=True, maximum=1.0)
    assert not strain_trust_filter_rejected(1.0, enabled=True, maximum=1.0)
    assert strain_trust_filter_rejected(1.01, enabled=True, maximum=1.0)
    assert strain_trust_filter_rejected(np.nan, enabled=True, maximum=1.0)
    assert not strain_trust_filter_rejected(np.nan, enabled=False, maximum=1.0)


def test_arap_constraint_is_zero_for_rigid_transform() -> None:
    inverse, volumes = tetrahedral_rest_data(REST, ELEMENT)
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transformed = REST @ rotation.T + np.asarray((2.0, -3.0, 0.4))
    constraints, gradients = arap_constraints_and_gradients_numpy(
        transformed, ELEMENT, inverse
    )
    assert volumes.tolist() == pytest.approx((1.0 / 6.0,))
    assert constraints.tolist() == pytest.approx((0.0,), abs=1.0e-12)
    assert gradients == pytest.approx(np.zeros((1, 4, 3)), abs=1.0e-12)


def test_arap_constraint_matches_singular_value_strain() -> None:
    inverse, _volumes = tetrahedral_rest_data(REST, ELEMENT)
    scaled = REST * np.asarray((1.2, 0.9, 1.0))
    constraints, _gradients = arap_constraints_and_gradients_numpy(
        scaled, ELEMENT, inverse
    )
    assert constraints.tolist() == pytest.approx((np.sqrt(0.2**2 + 0.1**2),))


def test_arap_constraint_penalizes_reflection() -> None:
    inverse, _volumes = tetrahedral_rest_data(REST, ELEMENT)
    reflected = REST * np.asarray((-1.0, 1.0, 1.0))
    constraints, gradients = arap_constraints_and_gradients_numpy(
        reflected, ELEMENT, inverse
    )
    assert constraints.tolist() == pytest.approx((2.0,))
    assert np.isfinite(gradients).all()


def test_arap_gradient_matches_finite_difference() -> None:
    inverse, _volumes = tetrahedral_rest_data(REST, ELEMENT)
    deformed = REST.copy()
    deformed[1] += (0.12, 0.03, -0.02)
    deformed[2] += (-0.01, -0.07, 0.04)
    constraints, gradients = arap_constraints_and_gradients_numpy(
        deformed, ELEMENT, inverse
    )
    step = 1.0e-6
    numerical = np.zeros((4, 3), dtype=np.float64)
    for vertex in range(4):
        for axis in range(3):
            positive = deformed.copy()
            negative = deformed.copy()
            positive[vertex, axis] += step
            negative[vertex, axis] -= step
            c_positive = arap_constraints_and_gradients_numpy(
                positive, ELEMENT, inverse
            )[0][0]
            c_negative = arap_constraints_and_gradients_numpy(
                negative, ELEMENT, inverse
            )[0][0]
            numerical[vertex, axis] = (c_positive - c_negative) / (2.0 * step)
    assert constraints[0] > 0.0
    assert gradients[0] == pytest.approx(numerical, abs=2.0e-6)
    assert gradients[0].sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-12)


def test_boundary_faces_remove_shared_tet_face() -> None:
    elements = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
    faces = boundary_faces(elements)
    assert faces.shape == (6, 3)
    assert not any(np.array_equal(np.sort(face), (0, 1, 2)) for face in faces)


def test_boundary_faces_point_away_from_positive_tet_interior() -> None:
    faces = boundary_faces(ELEMENT)
    tet_center = REST.mean(axis=0)
    for face in faces:
        triangle = REST[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        toward_interior = tet_center - triangle.mean(axis=0)
        assert float(np.dot(normal, toward_interior)) < 0.0


def test_triangle_contact_samples_cover_face_and_edges() -> None:
    faces = np.asarray(((2, 4, 6), (3, 5, 7)), dtype=np.int32)
    sample_faces, weights = triangle_contact_samples(faces)

    assert sample_faces.shape == (8, 3)
    assert weights.shape == (8, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        weights[:4],
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
        ),
    )
    np.testing.assert_array_equal(
        sample_faces[:4], np.tile(faces[0], (4, 1))
    )


def test_distal_finger_proxy_excludes_proximal_link_body() -> None:
    z = np.linspace(-0.10, 0.0, 21)
    vertices = np.asarray(
        [
            (x, y, value)
            for value in z
            for x in (-0.01, 0.01)
            for y in (-0.008, 0.008)
        ],
        dtype=np.float64,
    )
    selected = distal_finger_contact_vertex_indices(
        vertices, keep_fraction=0.60
    )
    proxy = vertices[selected]

    assert len(selected) >= 4
    assert float(np.max(proxy[:, 2])) <= -0.035
    assert float(np.min(proxy[:, 2])) == pytest.approx(-0.10)


def test_project_invokes_sqp_and_couples_fractional_multiplier_step() -> None:
    torch = pytest.importorskip("torch")

    config = MGPBDTetConfig(
        dt_s=0.01,
        shear_modulus_pa=1.0e4,
        relaxation=0.5,
        nonlinear_iterations=1,
        pcg_iterations=20,
        relative_residual=1.0e-5,
        line_search_enabled=True,
        line_search_objective="dual",
        line_search_acceptance_epsilon=0.0,
        line_search_minimum_step=1.0e-9,
        line_search_scale_lagrangian=True,
        line_search_rejected_lagrangian_policy="rollback",
        orientation_diagnostics_enabled=True,
        orientation_guard_enabled=True,
        orientation_guard_minimum_ratio=1.0e-6,
        strain_trust_filter_enabled=True,
        strain_trust_filter_maximum=1.0,
        sqp_direction_enabled=True,
        maximum_correction_m=None,
    )
    projector = VolumetricMGPBDProjector(
        torch.as_tensor(REST, dtype=torch.float32),
        ELEMENT,
        vertex_masses=np.ones(4),
        config=config,
    )
    calls: list[dict[str, object]] = []

    def fake_sqp(
        current,
        material_residual,
        constraints,
        gradients,
        base_direction,
        base_delta_lambda,
        diagonal,
    ):
        del current, constraints, diagonal
        linearized = (
            projector._local_rows_directional(gradients, base_direction)
            + projector.alpha_tilde * base_delta_lambda
        )
        slope = float(torch.dot(material_residual, linearized))
        assert slope < 0.0
        calls.append({"slope": slope})
        return (
            base_direction,
            base_delta_lambda,
            {"enabled": True, "mock_backend": True, "merit_slope": slope},
        )

    projector._sqp_constrained_direction = fake_sqp
    deformed = REST.astype(np.float32).copy()
    deformed[:, 1] *= 0.8
    projector.project(torch.as_tensor(deformed, dtype=torch.float32))

    assert len(calls) == 1
    record = projector.last_metrics["outer_iterations"][0]
    assert record["line_search_step"] == pytest.approx(0.5)
    assert record["lagrangian_acceptance_policy"] == (
        "accepted_scaled_trial_multiplier"
    )
    assert record["lagrangian_update_fraction"] == pytest.approx(0.5)
    assert record["lagrangian_fraction_matches_observed"]
    sqp = record["sqp_direction"]
    assert sqp["mock_backend"]
    assert sqp["coupled_position_multiplier_transaction"]
    assert sqp["accepted_step"] == pytest.approx(0.5)
    assert sqp["accepted_multiplier_fraction"] == pytest.approx(0.5)
    assert sqp["armijo_merit_after"] <= sqp["armijo_rhs"]
    assert sqp["armijo_satisfied"]
    assert sqp["rolled_back_atomically"]


def test_constrained_projector_rejects_post_commit_callback() -> None:
    torch = pytest.importorskip("torch")
    config = MGPBDTetConfig(
        dt_s=0.01,
        shear_modulus_pa=1.0e4,
        nonlinear_iterations=1,
        pcg_iterations=1,
        maximum_correction_m=None,
        line_search_enabled=True,
        line_search_objective="dual",
        line_search_scale_lagrangian=True,
        line_search_rejected_lagrangian_policy="rollback",
        orientation_diagnostics_enabled=True,
        orientation_guard_enabled=True,
        strain_trust_filter_enabled=True,
        strain_trust_filter_maximum=0.989,
        soc_admm_direction_enabled=True,
    )
    projector = VolumetricMGPBDProjector(
        torch.as_tensor(REST, dtype=torch.float32),
        ELEMENT,
        vertex_masses=np.ones(4),
        config=config,
    )
    with pytest.raises(ValueError, match="post-iteration callback"):
        projector.project(
            torch.as_tensor(REST, dtype=torch.float32),
            post_iteration=lambda positions: positions,
        )


def _soc_enabled_projector_config() -> MGPBDTetConfig:
    return MGPBDTetConfig(
        dt_s=0.01,
        shear_modulus_pa=1.0e4,
        relaxation=1.0,
        nonlinear_iterations=1,
        pcg_iterations=20,
        relative_residual=1.0e-5,
        line_search_enabled=True,
        line_search_objective="dual",
        line_search_acceptance_epsilon=0.0,
        line_search_minimum_step=1.0e-9,
        line_search_scale_lagrangian=True,
        line_search_rejected_lagrangian_policy="rollback",
        orientation_diagnostics_enabled=True,
        orientation_guard_enabled=True,
        orientation_guard_minimum_ratio=1.0e-6,
        strain_trust_filter_enabled=True,
        strain_trust_filter_maximum=0.989,
        soc_admm_direction_enabled=True,
        maximum_correction_m=None,
    )


def test_project_soc_admm_bypasses_legacy_direction_and_records_transaction(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    from sim.genesis_so101 import mgpbd_soc_admm

    config = _soc_enabled_projector_config()
    projector = VolumetricMGPBDProjector(
        torch.as_tensor(REST, dtype=torch.float32),
        ELEMENT,
        vertex_masses=np.ones(4),
        config=config,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy PCG/AMG/SQP path must be bypassed")

    projector._pcg = forbidden
    projector._assemble_cpu_matrix = forbidden
    projector._sqp_constrained_direction = forbidden
    captured: dict[str, object] = {}

    def fake_soc_admm_direction(**kwargs):
        captured.update(kwargs)
        direction = projector.rest_positions - kwargs["current"]
        linearized = projector._local_rows_directional(
            kwargs["material_gradients"], direction
        )
        delta_lambda = -(kwargs["q"] + linearized) / kwargs["alpha"]
        return mgpbd_soc_admm.SOCADMMDirectionResult(
            direction=direction,
            delta_lambda=delta_lambda,
            metrics={
                "schema_version": (
                    "radeon_oneloop.mgpbd_soc_admm_direction.v1"
                ),
                "backend": "mock_matrix_free_soc_admm",
                "converged": True,
                "passed": True,
                "failure": None,
                "admm_iterations": 7,
                "pcg_iterations_total": 11,
                "checks": {"mock_gate": True},
            },
        )

    monkeypatch.setattr(
        mgpbd_soc_admm,
        "solve_soc_admm_direction",
        fake_soc_admm_direction,
    )
    deformed = REST.astype(np.float32).copy()
    deformed[:, 1] *= 0.8
    projected = projector.project(torch.as_tensor(deformed))

    torch.testing.assert_close(projected, projector.rest_positions)
    torch.testing.assert_close(
        captured["masses"], torch.reciprocal(projector.inverse_mass)
    )
    initial_constraints, initial_gradients, _active = (
        projector.constraints_and_gradients(torch.as_tensor(deformed))
    )
    torch.testing.assert_close(captured["q"], initial_constraints)
    torch.testing.assert_close(captured["material_gradients"], initial_gradients)
    assert captured["alpha"] is projector.alpha_tilde
    assert captured["config"] == config.soc_admm_config()

    metrics = projector.last_metrics
    assert metrics["direction_backend"] == "soc_admm"
    assert metrics["legacy_direction_pcg_skipped"]
    assert metrics["pcg_iterations_total"] == 0
    assert metrics["soc_admm_iterations_total"] == 7
    assert metrics["soc_admm_pcg_iterations_total"] == 11
    assert metrics["amg_hierarchy_builder"] == "not_built"
    record = metrics["outer_iterations"][0]
    assert record["direction_backend"] == "soc_admm"
    assert record["legacy_direction_pcg_skipped"]
    assert record["pcg_iterations"] == 0
    assert record["fine_diagonal_sum"] is None
    assert record["correction_global_scale"] == pytest.approx(1.0)
    soc = record["soc_admm_direction"]
    assert soc["backend"] == "mock_matrix_free_soc_admm"
    assert soc["coupled_position_multiplier_transaction"]
    assert soc["accepted_step"] == pytest.approx(1.0)
    assert soc["accepted_multiplier_fraction"] == pytest.approx(1.0)
    assert soc["armijo_satisfied"]
    assert soc["rolled_back_atomically"]


def test_project_soc_admm_failure_preserves_receipt_and_commits_nothing(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    from sim.genesis_so101 import mgpbd_soc_admm

    projector = VolumetricMGPBDProjector(
        torch.as_tensor(REST, dtype=torch.float32),
        ELEMENT,
        vertex_masses=np.ones(4),
        config=_soc_enabled_projector_config(),
    )
    projector._pcg = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("legacy PCG must be bypassed")
    )
    receipt = {
        "schema_version": "radeon_oneloop.mgpbd_soc_admm_direction.v1",
        "converged": False,
        "failure": "test_failure",
    }

    def fail_soc(**_kwargs):
        raise mgpbd_soc_admm.SOCADMMConvergenceError("test failure", receipt)

    monkeypatch.setattr(mgpbd_soc_admm, "solve_soc_admm_direction", fail_soc)
    deformed = torch.as_tensor(
        REST.astype(np.float32) * np.asarray((1.0, 0.8, 1.0), dtype=np.float32)
    )
    snapshot = deformed.clone()
    with pytest.raises(mgpbd_soc_admm.SOCADMMConvergenceError) as caught:
        projector.project(deformed)

    assert caught.value.receipt is receipt
    torch.testing.assert_close(deformed, snapshot)
    assert projector.last_metrics["projection_failed"]
    assert projector.last_metrics["failure"] == "soc_admm_direction_failed"
    assert projector.last_metrics["soc_admm_direction"] is receipt
    assert projector.last_metrics["legacy_direction_pcg_skipped"]
    assert projector.last_metrics["projected_frames"] == 0
    assert projector.last_metrics["last_accepted_outer_iteration"] == 0
    assert projector.last_accepted_outer_iteration == 0
    torch.testing.assert_close(
        projector.last_accepted_outer_positions,
        snapshot,
    )
