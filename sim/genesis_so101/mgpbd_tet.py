"""MGPBD volumetric plush solver for the TRELLIS.2 handover object.

This is a clean-room Radeon/PyTorch implementation of the algorithm described
in Li et al., *MGPBD: Multigrid Preconditioned Position Based Dynamics* and
checked against the authors' public ``engine/soft/soft3d.py`` implementation.
The physical unknown is the tetrahedral proxy used by Genesis, with exactly
one ARAP constraint per tetrahedron::

    C_t(x) = sqrt(sum_i (sigma_i(F_t) - 1)^2)
    A      = J M^-1 J^T + alpha_tilde

``A delta_lambda = -C - alpha_tilde lambda`` is solved globally with PCG and
an unsmoothed-aggregation multigrid V-cycle.  This is deliberately not the old
six-edge projection prototype and it does not leave Genesis FEM responsible
for the constitutive dynamics.

Genesis still owns the two rigid SO-101 arms and rendering.  The adapter at
the end of this module exposes gripper convex hulls and the visual tetrahedral
binding through the same narrow interface as the existing custom XPBD solver.
Heavy Torch, SciPy, TetGen, and Trimesh imports remain lazy so topology and
formula tests can run on the lightweight orchestration host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import time

import numpy as np


@dataclass(frozen=True)
class MGPBDTetConfig:
    """Paper-aligned ARAP/MGPCG and live-contact settings."""

    dt_s: float = 1.0 / 120.0
    # E=40 kPa and nu=0.34 from the accepted FEM baseline imply mu~=14.9 kPa.
    shear_modulus_pa: float = 1.5e4
    relaxation: float = 0.10
    # Live MGPBD uses a per-frame time budget: one nonlinear update and a
    # stronger MGPCG solve, then continues convergence on the next 120 Hz
    # state.  Offline comparisons may override these values.
    nonlinear_iterations: int = 1
    pcg_iterations: int = 20
    relative_residual: float = 1.0e-3
    outer_absolute_residual: float | None = None
    outer_relative_residual: float | None = None
    constraint_epsilon: float = 1.0e-6
    amg_strength_threshold: float = 0.10
    amg_coarsest_size: int = 128
    amg_max_levels: int = 5
    amg_setup_interval_frames: int = 10_000
    amg_min_active_fraction: float = 0.25
    smoother_iterations: int = 2
    # The equilibrated proxy dual matrix has measured rho(D^-1 A)=8.50; the
    # authors' 1/rho Jacobi rule therefore gives approximately 0.1.
    smoother_weight: float = 0.10
    smoother_weight_mode: str = "fixed"
    amg_hierarchy_mode: str = "auto"
    symmetric_diagonal_equilibration: bool = True
    line_search_enabled: bool = True
    line_search_objective: str = "constraint"
    # Public-code fidelity uses a strict comparator and a 1e-9 minimum step.
    # ``None`` preserves the historical live cutoff of relaxation / 4096.
    line_search_acceptance_epsilon: float = 1.0e-7
    line_search_minimum_step: float | None = None
    # The public implementation updates the full multiplier before searching
    # only the position step.  Scaling both is retained as an ablation; the
    # orientation-safe contract keeps public accepted-step semantics and only
    # rolls both fields back when no feasible step exists.
    line_search_scale_lagrangian: bool = False
    line_search_rejected_lagrangian_policy: str = "rollback"
    orientation_diagnostics_enabled: bool = False
    orientation_guard_enabled: bool = False
    orientation_guard_minimum_ratio: float = 1.0e-6
    # Downstream recovery-only trust filter.  It is disabled by default and in
    # the official-fidelity profile; it is not part of public MGPBD.
    strain_trust_filter_enabled: bool = False
    strain_trust_filter_maximum: float = 1.0
    # Opt-in downstream SQP direction.  It preserves the MGPBD Gauss--Newton
    # Hessian and uses a small active-set Schur complement on the current A.
    sqp_direction_enabled: bool = False
    sqp_strain_activation_threshold: float = 0.98
    sqp_strain_maximum: float = 1.0
    sqp_minimum_signed_volume_ratio: float = 1.0e-6
    sqp_determinant_activation_ratio: float = 5.0e-3
    sqp_strain_fraction_to_boundary: float = 0.8
    sqp_volume_fraction_to_boundary: float = 0.8
    sqp_primal_tolerance: float = 2.0e-5
    sqp_dual_tolerance: float = 1.0e-8
    sqp_kkt_relative_tolerance: float = 5.0e-4
    sqp_coupled_relative_tolerance: float = 5.0e-5
    # The auxiliary A^{-1} columns are more sensitive than the material solve:
    # their dual residual is mapped through J^T alpha^{-1} in the SQP
    # stationarity equation.  Iterative refinement prevents a nominal 1e-5
    # MGPCG residual from being amplified into a percent-level KKT defect.
    sqp_auxiliary_relative_residual: float = 1.0e-6
    sqp_maximum_auxiliary_refinements: int = 3
    sqp_maximum_active_set_iterations: int = 2_048
    sqp_maximum_active_constraints: int = 1_024
    sqp_maximum_nonlinear_cut_resolves: int = 64
    sqp_maximum_cuts_per_resolve: int = 64
    sqp_boundary_scan_intervals: int = 32
    sqp_boundary_bisection_iterations: int = 40
    sqp_armijo_coefficient: float = 1.0e-4
    # Opt-in replacement for the non-scalable explicit active-set/SQP
    # direction.  These fields mirror ``SOCADMMConfig`` deliberately: a
    # conformance receipt can therefore reconstruct the exact inner solve
    # without relying on module defaults.  The feature remains disabled for
    # every existing live/default profile.
    soc_admm_direction_enabled: bool = False
    soc_admm_beta: float = 1.0e-3
    soc_admm_scale_beta_by_operator_diagonal: bool = False
    soc_admm_beta_minimum: float = 1.0e-4
    soc_admm_beta_maximum: float = 1.0
    soc_admm_kkt_polish_beta_maximum: float | None = None
    soc_admm_adaptive_beta: bool = True
    soc_admm_beta_update_interval: int = 25
    soc_admm_beta_balance_ratio: float = 5.0
    soc_admm_beta_update_factor: float = 2.0
    soc_admm_work_radius: float = 0.989
    soc_admm_true_arap_maximum: float = 1.0
    soc_admm_minimum_signed_volume_ratio: float = 1.0e-6
    soc_admm_maximum_iterations: int = 2_000
    soc_admm_required_consecutive_gate_passes: int = 3
    soc_admm_primal_tolerance: float = 2.0e-4
    soc_admm_dual_relative_tolerance: float = 5.0e-4
    soc_admm_stationarity_relative_tolerance: float = 5.0e-4
    soc_admm_accepted_dtype_stationarity_safety_factor: float = 0.99
    soc_admm_normal_cone_tolerance: float = 2.0e-5
    soc_admm_coupled_material_relative_tolerance: float = 2.0e-6
    soc_admm_pcg_maximum_iterations: int = 500
    soc_admm_pcg_relative_tolerance: float = 5.0e-6
    soc_admm_pcg_absolute_tolerance: float = 1.0e-10
    soc_admm_pcg_residual_replacement_interval: int = 25
    damping_retention: float = 0.965
    particle_radius_m: float = 0.0020
    table_friction: float = 0.85
    gripper_friction: float = 1.15
    contact_slop_m: float = 0.00035
    contact_release_m: float = 0.0015
    two_finger_coarse_transfer_gain: float = 1.0
    two_finger_transfer_closure_threshold: float = 0.80
    maximum_grasp_transport_m: float = 0.012
    grasp_contact_persistence_frames: int = 8
    minimum_signed_volume_ratio: float = 0.0
    collision_passes: int = 2
    maximum_correction_m: float | None = 0.012
    maximum_sweep_margin_m: float = 0.003

    def validate(self) -> None:
        if self.dt_s <= 0.0 or self.shear_modulus_pa <= 0.0:
            raise ValueError("MGPBD timestep and shear modulus must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("MGPBD relaxation must be in (0, 1]")
        if self.nonlinear_iterations < 1 or self.pcg_iterations < 1:
            raise ValueError("MGPBD iteration counts must be positive")
        if not 0.0 < self.relative_residual < 1.0:
            raise ValueError("MGPBD relative residual must be in (0, 1)")
        if self.outer_absolute_residual is not None and self.outer_absolute_residual <= 0.0:
            raise ValueError("MGPBD outer absolute residual must be positive")
        if (
            self.outer_relative_residual is not None
            and not 0.0 < self.outer_relative_residual < 1.0
        ):
            raise ValueError("MGPBD outer relative residual must be in (0, 1)")
        if self.constraint_epsilon <= 0.0:
            raise ValueError("MGPBD constraint epsilon must be positive")
        if not 0.0 <= self.amg_strength_threshold <= 1.0:
            raise ValueError("AMG strength threshold must be in [0, 1]")
        if self.amg_coarsest_size < 2 or self.amg_max_levels < 1:
            raise ValueError("invalid AMG hierarchy limits")
        if self.amg_setup_interval_frames < 1:
            raise ValueError("AMG setup interval must be positive")
        if not 0.0 < self.amg_min_active_fraction <= 1.0:
            raise ValueError("AMG active fraction must be in (0, 1]")
        if self.smoother_iterations < 1 or not 0.0 < self.smoother_weight < 2.0:
            raise ValueError("invalid AMG smoother settings")
        if self.smoother_weight_mode not in {"fixed", "fine_spectral_radius"}:
            raise ValueError("invalid AMG smoother weight mode")
        if self.amg_hierarchy_mode not in {"auto", "matrix_ua", "topology_ua"}:
            raise ValueError("invalid MGPBD AMG hierarchy mode")
        if self.line_search_objective not in {"constraint", "dual"}:
            raise ValueError("invalid MGPBD line-search objective")
        if self.line_search_acceptance_epsilon < 0.0:
            raise ValueError("line-search acceptance epsilon must be nonnegative")
        if not np.isfinite(self.line_search_acceptance_epsilon):
            raise ValueError("line-search acceptance epsilon must be finite")
        if self.line_search_rejected_lagrangian_policy not in {
            "retain_full",
            "rollback",
        }:
            raise ValueError("invalid rejected line-search multiplier policy")
        if (
            self.line_search_minimum_step is not None
            and (
                not np.isfinite(self.line_search_minimum_step)
                or not 0.0 < self.line_search_minimum_step < self.relaxation
            )
        ):
            raise ValueError(
                "line-search minimum step must be finite and in (0, relaxation)"
            )
        if (
            not np.isfinite(self.orientation_guard_minimum_ratio)
            or not 0.0 < self.orientation_guard_minimum_ratio < 1.0
        ):
            raise ValueError("orientation guard ratio must be in (0, 1)")
        if self.orientation_guard_enabled and (
            not self.orientation_diagnostics_enabled
            or not self.line_search_enabled
        ):
            raise ValueError(
                "orientation guard requires diagnostics and line search"
            )
        if (
            not np.isfinite(self.strain_trust_filter_maximum)
            or self.strain_trust_filter_maximum <= 0.0
        ):
            raise ValueError("strain trust-filter maximum must be finite and positive")
        if self.strain_trust_filter_enabled and not self.line_search_enabled:
            raise ValueError("strain trust filter requires line search")
        sqp_scalars = np.asarray(
            (
                self.sqp_strain_activation_threshold,
                self.sqp_strain_maximum,
                self.sqp_minimum_signed_volume_ratio,
                self.sqp_determinant_activation_ratio,
                self.sqp_strain_fraction_to_boundary,
                self.sqp_volume_fraction_to_boundary,
                self.sqp_primal_tolerance,
                self.sqp_dual_tolerance,
                self.sqp_kkt_relative_tolerance,
                self.sqp_coupled_relative_tolerance,
                self.sqp_auxiliary_relative_residual,
                self.sqp_armijo_coefficient,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(sqp_scalars).all():
            raise ValueError("SQP direction configuration must be finite")
        if not (
            0.0
            < self.sqp_strain_activation_threshold
            < self.sqp_strain_maximum
        ):
            raise ValueError("invalid SQP strain activation/maximum")
        if not (
            0.0
            <= self.sqp_minimum_signed_volume_ratio
            < self.sqp_determinant_activation_ratio
        ):
            raise ValueError("invalid SQP determinant activation range")
        if not 0.0 < self.sqp_strain_fraction_to_boundary < 1.0:
            raise ValueError("invalid SQP strain fraction-to-boundary")
        if not 0.0 < self.sqp_volume_fraction_to_boundary < 1.0:
            raise ValueError("invalid SQP volume fraction-to-boundary")
        if (
            self.sqp_primal_tolerance <= 0.0
            or self.sqp_dual_tolerance < 0.0
            or self.sqp_kkt_relative_tolerance <= 0.0
            or self.sqp_coupled_relative_tolerance <= 0.0
            or self.sqp_auxiliary_relative_residual <= 0.0
        ):
            raise ValueError("invalid SQP KKT tolerances")
        if (
            self.sqp_maximum_active_set_iterations < 1
            or self.sqp_maximum_active_constraints < 1
            or self.sqp_maximum_nonlinear_cut_resolves < 1
            or self.sqp_maximum_cuts_per_resolve < 1
            or self.sqp_maximum_auxiliary_refinements < 0
            or self.sqp_boundary_scan_intervals < 1
            or self.sqp_boundary_bisection_iterations < 1
        ):
            raise ValueError("invalid SQP active-set/cutting-plane limits")
        if not 0.0 < self.sqp_armijo_coefficient < 0.5:
            raise ValueError("invalid SQP Armijo coefficient")
        if self.sqp_direction_enabled and self.soc_admm_direction_enabled:
            raise ValueError(
                "legacy SQP and SOC-ADMM directions are mutually exclusive"
            )
        if self.sqp_direction_enabled and (
            not self.line_search_enabled
            or self.line_search_objective != "dual"
            or not self.line_search_scale_lagrangian
            or self.line_search_rejected_lagrangian_policy != "rollback"
            or not self.orientation_guard_enabled
            or not self.strain_trust_filter_enabled
            or self.maximum_correction_m is not None
        ):
            raise ValueError(
                "SQP direction requires coupled rollback line search, exact "
                "orientation/strain guards, and no post-direction clip"
            )
        if self.soc_admm_direction_enabled and (
            not self.line_search_enabled
            or self.line_search_objective != "dual"
            or not self.line_search_scale_lagrangian
            or self.line_search_rejected_lagrangian_policy != "rollback"
            or not self.orientation_guard_enabled
            or not self.strain_trust_filter_enabled
            or self.maximum_correction_m is not None
        ):
            raise ValueError(
                "SOC-ADMM direction requires coupled rollback line search, "
                "exact orientation/strain guards, and no post-direction clip"
            )
        if (
            self.soc_admm_direction_enabled
            and self.strain_trust_filter_maximum
            > self.soc_admm_work_radius
        ):
            raise ValueError(
                "SOC-ADMM outer accepted-state strain gate must not exceed "
                "the next direction's work radius"
            )
        # Delegate the numerical/proof-margin checks to the authoritative
        # inner-solver configuration.  This import is Torch-free and keeps
        # duplicated validation logic out of the projector.
        self.soc_admm_config().validate()
        if not 0.0 <= self.damping_retention <= 1.0:
            raise ValueError("MGPBD damping retention must be in [0, 1]")
        if self.particle_radius_m <= 0.0:
            raise ValueError("MGPBD contact radius must be positive")
        if self.table_friction < 0.0 or self.gripper_friction < 0.0:
            raise ValueError("MGPBD friction must be nonnegative")
        if self.contact_slop_m < 0.0:
            raise ValueError("MGPBD contact slop must be nonnegative")
        if self.contact_release_m <= self.contact_slop_m:
            raise ValueError("MGPBD contact release must exceed contact slop")
        if not 0.0 <= self.two_finger_coarse_transfer_gain <= 1.0:
            raise ValueError("MGPBD coarse contact transfer gain must be in [0, 1]")
        if not 0.0 <= self.two_finger_transfer_closure_threshold <= 1.0:
            raise ValueError("MGPBD transfer closure threshold must be in [0, 1]")
        if self.maximum_grasp_transport_m <= 0.0:
            raise ValueError("MGPBD maximum grasp transport must be positive")
        if self.grasp_contact_persistence_frames < 0:
            raise ValueError("MGPBD grasp persistence must be nonnegative")
        if not 0.0 <= self.minimum_signed_volume_ratio < 1.0:
            raise ValueError("MGPBD minimum signed volume ratio must be in [0, 1)")
        if (
            self.collision_passes < 1
            or (
                self.maximum_correction_m is not None
                and self.maximum_correction_m <= 0.0
            )
            or self.maximum_sweep_margin_m < 0.0
        ):
            raise ValueError("invalid MGPBD collision settings")

    def alpha_tilde(self, rest_volumes_m3: np.ndarray) -> np.ndarray:
        """Return official ``1 / (mu * volume * dt^2)`` compliance."""

        volumes = np.asarray(rest_volumes_m3, dtype=np.float64).reshape(-1)
        if not len(volumes) or np.any(volumes <= 0.0):
            raise ValueError("MGPBD rest volumes must be positive")
        return 1.0 / (
            self.shear_modulus_pa * volumes * self.dt_s * self.dt_s
        )

    def soc_admm_config(self):
        """Materialize the exact matrix-free SOC direction contract."""

        from sim.genesis_so101.mgpbd_soc_admm import SOCADMMConfig

        return SOCADMMConfig(
            beta=self.soc_admm_beta,
            scale_beta_by_operator_diagonal=(
                self.soc_admm_scale_beta_by_operator_diagonal
            ),
            beta_minimum=self.soc_admm_beta_minimum,
            beta_maximum=self.soc_admm_beta_maximum,
            kkt_polish_beta_maximum=(
                self.soc_admm_kkt_polish_beta_maximum
            ),
            adaptive_beta=self.soc_admm_adaptive_beta,
            beta_update_interval=self.soc_admm_beta_update_interval,
            beta_balance_ratio=self.soc_admm_beta_balance_ratio,
            beta_update_factor=self.soc_admm_beta_update_factor,
            work_radius=self.soc_admm_work_radius,
            true_arap_maximum=self.soc_admm_true_arap_maximum,
            minimum_signed_volume_ratio=(
                self.soc_admm_minimum_signed_volume_ratio
            ),
            maximum_admm_iterations=self.soc_admm_maximum_iterations,
            required_consecutive_gate_passes=(
                self.soc_admm_required_consecutive_gate_passes
            ),
            admm_primal_tolerance=self.soc_admm_primal_tolerance,
            admm_dual_relative_tolerance=(
                self.soc_admm_dual_relative_tolerance
            ),
            stationarity_relative_tolerance=(
                self.soc_admm_stationarity_relative_tolerance
            ),
            accepted_dtype_stationarity_safety_factor=(
                self.soc_admm_accepted_dtype_stationarity_safety_factor
            ),
            normal_cone_tolerance=self.soc_admm_normal_cone_tolerance,
            coupled_material_relative_tolerance=(
                self.soc_admm_coupled_material_relative_tolerance
            ),
            pcg_maximum_iterations=self.soc_admm_pcg_maximum_iterations,
            pcg_relative_tolerance=self.soc_admm_pcg_relative_tolerance,
            pcg_absolute_tolerance=self.soc_admm_pcg_absolute_tolerance,
            pcg_residual_replacement_interval=(
                self.soc_admm_pcg_residual_replacement_interval
            ),
        )

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        self.validate()
        return asdict(self)


def line_search_objective_rejected(
    candidate: float, current: float, acceptance_epsilon: float
) -> bool:
    """Apply the configured strict or legacy-tolerant merit comparator."""

    if not np.isfinite(acceptance_epsilon) or acceptance_epsilon < 0.0:
        raise ValueError(
            "line-search acceptance epsilon must be finite and nonnegative"
        )
    # A non-finite merit value is never evidence of descent.  Treating NaN as
    # accepted here silently poisons both state and the conformance record.
    if not np.isfinite(candidate) or not np.isfinite(current):
        return True
    return (
        candidate >= current
        if acceptance_epsilon == 0.0
        else candidate > current + acceptance_epsilon
    )


def armijo_merit_rejected(
    candidate_norm: float,
    current_norm: float,
    step: float,
    slope: float,
    coefficient: float,
) -> bool:
    """Return whether a coupled SQP trial fails the squared-dual Armijo test."""

    values = np.asarray(
        (candidate_norm, current_norm, step, slope, coefficient),
        dtype=np.float64,
    )
    if (
        not np.isfinite(values).all()
        or candidate_norm < 0.0
        or current_norm < 0.0
        or step < 0.0
        or slope > 0.0
        or not 0.0 < coefficient < 0.5
    ):
        return True
    # A zero slope is valid only for the exact stationary no-op.  Treating any
    # other flat direction as Armijo descent would hide a stalled constrained
    # solve, while rejecting (0 -> 0) makes an already solved rest state fail.
    if slope == 0.0 and (candidate_norm != 0.0 or current_norm != 0.0):
        return True
    candidate_merit = 0.5 * candidate_norm * candidate_norm
    current_merit = 0.5 * current_norm * current_norm
    return bool(
        candidate_merit
        > current_merit + coefficient * step * slope
    )


def strain_trust_filter_rejected(
    candidate_maximum: float,
    *,
    enabled: bool,
    maximum: float,
) -> bool:
    """Fail closed when a downstream trial exits its ARAP strain trust region."""

    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("strain trust-filter maximum must be finite and positive")
    return bool(
        enabled
        and (
            not np.isfinite(candidate_maximum)
            or candidate_maximum > maximum
        )
    )


def lagrangian_acceptance_policy(
    *,
    rejected: bool,
    rejected_policy: str,
    line_search_scale_lagrangian: bool,
    accepted_step: float = 1.0,
    correction_global_scale: float = 1.0,
) -> tuple[str, float]:
    """Return the named multiplier transaction and its exact scalar fraction."""

    if rejected_policy not in {"retain_full", "rollback"}:
        raise ValueError("invalid rejected line-search multiplier policy")
    if rejected:
        if rejected_policy == "retain_full":
            return "retain_full_public_multiplier", 1.0
        return "rollback_multiplier_with_position", 0.0
    if line_search_scale_lagrangian:
        fraction = accepted_step * correction_global_scale
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("invalid scaled multiplier fraction")
        return "accepted_scaled_trial_multiplier", float(fraction)
    return "accepted_full_trial_multiplier", 1.0


def unique_tet_edges(elements: np.ndarray) -> np.ndarray:
    """Return unique edges for diagnostics only, never as MGPBD constraints."""

    elements = np.asarray(elements, dtype=np.int64).reshape(-1, 4)
    if not len(elements) or int(elements.min()) < 0:
        raise ValueError("tet elements must be non-empty and nonnegative")
    local_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64
    )
    edges = elements[:, local_pairs].reshape(-1, 2)
    edges.sort(axis=1)
    return np.unique(edges, axis=0).astype(np.int32)


def boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Return outward boundary triangles for positively oriented tetrahedra."""

    elements = np.asarray(elements, dtype=np.int64).reshape(-1, 4)
    # For det([x1-x0, x2-x0, x3-x0]) > 0 these wind away from the
    # opposite vertex.  The previous ordering was consistently inward; that
    # did not affect topology counts but produced reversed physical OBJ
    # normals and made visual integrity harder to judge.
    local_faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    faces = elements[:, local_faces].reshape(-1, 3)
    canonical = np.sort(faces, axis=1)
    unique, first, counts = np.unique(
        canonical, axis=0, return_index=True, return_counts=True
    )
    del unique
    return faces[first[counts == 1]].astype(np.int32)


def triangle_contact_samples(
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense face/edge collision samples without adding physical DOFs."""

    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles) or int(triangles.min()) < 0:
        raise ValueError("contact sample faces must be non-empty and nonnegative")
    weights_per_face = np.asarray(
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
        ),
        dtype=np.float32,
    )
    sample_faces = np.repeat(triangles, len(weights_per_face), axis=0)
    sample_weights = np.tile(weights_per_face, (len(triangles), 1))
    return sample_faces.astype(np.int32), sample_weights


def distal_finger_contact_vertex_indices(
    local_vertices: np.ndarray, *, keep_fraction: float = 0.65
) -> np.ndarray:
    """Select the distal straight section of a gripper mesh.

    The SO-101 fixed follower and moving jaw are concave articulated parts.
    Taking one convex hull of the complete link fills the open jaw cavity and
    reports contact before the rendered fingers touch.  Their contact-bearing
    distal sections are approximately straight, so retain the far end along
    the principal axis and convexify only that pad section.
    """

    vertices = np.asarray(local_vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertices) < 8 or not np.isfinite(vertices).all():
        raise ValueError("finger contact mesh must contain finite vertices")
    if not 0.25 <= keep_fraction <= 0.90:
        raise ValueError("finger distal keep fraction must be in [0.25, 0.90]")
    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered / float(len(vertices))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection = vertices @ axis
    low = float(np.min(projection))
    high = float(np.max(projection))
    span = high - low
    if span <= 1.0e-6:
        raise ValueError("finger contact mesh principal extent is degenerate")
    endpoint_width = 0.05 * span
    low_center = vertices[projection <= low + endpoint_width].mean(axis=0)
    high_center = vertices[projection >= high - endpoint_width].mean(axis=0)
    high_is_distal = float(np.linalg.norm(high_center)) >= float(
        np.linalg.norm(low_center)
    )
    if high_is_distal:
        cutoff = high - keep_fraction * span
        selected = np.flatnonzero(projection >= cutoff)
    else:
        cutoff = low + keep_fraction * span
        selected = np.flatnonzero(projection <= cutoff)
    if len(selected) < 4:
        raise ValueError("finger distal proxy contains too few vertices")
    return selected.astype(np.int64)


def tetrahedral_rest_data(
    rest_positions: np.ndarray, elements: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Dm^-1`` and positive rest volumes for each tetrahedron."""

    positions = np.asarray(rest_positions, dtype=np.float64).reshape(-1, 3)
    elements = np.asarray(elements, dtype=np.int64).reshape(-1, 4)
    if not len(positions) or not len(elements):
        raise ValueError("tet rest mesh must be non-empty")
    if int(elements.min()) < 0 or int(elements.max()) >= len(positions):
        raise ValueError("tet element index lies outside rest positions")
    tets = positions[elements]
    dm = np.stack(
        (tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]),
        axis=-1,
    )
    determinants = np.linalg.det(dm)
    volumes = np.abs(determinants) / 6.0
    if np.any(volumes <= 1.0e-14):
        raise ValueError("tet rest mesh contains a degenerate element")
    return np.linalg.inv(dm), volumes


def load_precomputed_tet_mesh(
    path: Path,
    manifest_path: Path,
    *,
    expected_contact_radius_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load and independently validate a quality-gated dense tet artifact."""

    path = Path(path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"dense MGPBD volume or manifest is missing: {path}, {manifest_path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "radeon_oneloop.mgpbd_dense_volume.v1":
        raise ValueError("unsupported dense MGPBD volume manifest")
    if not manifest.get("quality_gate", {}).get("passed", False):
        raise ValueError("dense MGPBD volume did not pass its quality gate")
    if digest.hexdigest() != manifest.get("volume", {}).get("sha256"):
        raise ValueError("dense MGPBD volume hash does not match its manifest")
    with np.load(path, allow_pickle=False) as archive:
        if not {"nodes", "elements", "boundary_faces"}.issubset(archive.files):
            raise ValueError("dense MGPBD archive is missing required arrays")
        nodes = np.asarray(archive["nodes"], dtype=np.float32).reshape(-1, 3)
        elements = np.asarray(archive["elements"], dtype=np.int32).reshape(-1, 4)
        stored_boundary = np.asarray(
            archive["boundary_faces"], dtype=np.int32
        ).reshape(-1, 3)
    if not np.isfinite(nodes).all():
        raise ValueError("dense MGPBD nodes must be finite")
    inverse, volumes = tetrahedral_rest_data(nodes, elements)
    tetrahedra = nodes[elements].astype(np.float64)
    signed_six = np.einsum(
        "ij,ij->i",
        np.cross(
            tetrahedra[:, 1] - tetrahedra[:, 0],
            tetrahedra[:, 2] - tetrahedra[:, 0],
        ),
        tetrahedra[:, 3] - tetrahedra[:, 0],
    )
    if np.any(signed_six <= 0.0):
        raise ValueError("dense MGPBD tetrahedra must have positive orientation")
    actual_boundary = boundary_faces(elements)
    if not np.array_equal(
        np.unique(np.sort(stored_boundary, axis=1), axis=0),
        np.unique(np.sort(actual_boundary, axis=1), axis=0),
    ):
        raise ValueError("dense MGPBD stored boundary does not match its tets")
    volume_manifest = manifest["volume"]
    if int(volume_manifest.get("nodes", -1)) != len(nodes) or int(
        volume_manifest.get("tetrahedra", -1)
    ) != len(elements):
        raise ValueError("dense MGPBD array counts do not match the manifest")
    runtime = manifest.get("runtime", {})
    contact_radius = float(runtime.get("contact_radius_m", -1.0))
    if (
        expected_contact_radius_m is not None
        and not np.isclose(
            contact_radius,
            expected_contact_radius_m,
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError(
            "dense MGPBD contact radius does not match the solver configuration"
        )
    conditions = np.linalg.cond(np.linalg.inv(inverse))
    return nodes, elements, {
        "method": "precomputed_isotropic_surface_tetgen_dense_volume",
        "volume_path": str(path),
        "manifest_path": str(manifest_path),
        "sha256": digest.hexdigest(),
        "physics_vertices": int(len(nodes)),
        "physics_tetrahedra": int(len(elements)),
        "boundary_vertices": int(len(np.unique(actual_boundary))),
        "boundary_faces": int(len(actual_boundary)),
        "rest_volume_m3": float(np.sum(volumes)),
        "minimum_tet_volume_m3": float(np.min(volumes)),
        "maximum_rest_matrix_condition": float(np.max(conditions)),
        "contact_radius_m": contact_radius,
        "visual_binding": runtime.get("visual_binding"),
        "quality_gate": dict(manifest["quality_gate"]),
        "source_manifest": manifest,
    }


def lumped_vertex_masses(
    elements: np.ndarray, rest_volumes_m3: np.ndarray, total_mass_kg: float, vertex_count: int
) -> np.ndarray:
    """Distribute volume-proportional tet mass equally over its four vertices."""

    if total_mass_kg <= 0.0 or vertex_count < 4:
        raise ValueError("MGPBD mass and vertex count must be positive")
    elements = np.asarray(elements, dtype=np.int64).reshape(-1, 4)
    volumes = np.asarray(rest_volumes_m3, dtype=np.float64).reshape(-1)
    if len(elements) != len(volumes) or np.any(volumes <= 0.0):
        raise ValueError("tet volume count mismatch")
    density = total_mass_kg / float(np.sum(volumes))
    masses = np.zeros(vertex_count, dtype=np.float64)
    contribution = density * volumes / 4.0
    for local_index in range(4):
        np.add.at(masses, elements[:, local_index], contribution)
    if np.any(masses <= 0.0):
        raise ValueError("MGPBD tet mesh contains an unreferenced vertex")
    return masses


def resolve_vertex_masses(
    elements: np.ndarray,
    rest_volumes: np.ndarray,
    vertex_count: int,
    *,
    total_mass: float | None,
    explicit_vertex_masses: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Resolve either the live SI lumped mass or an explicit reference mass.

    The public bunny scene uses one simulation mass unit per vertex.  The live
    doll instead distributes a measured total mass by tetrahedral volume.
    Keeping the two modes explicit prevents a reference test from silently
    changing the robot-scene material convention.
    """

    if explicit_vertex_masses is not None:
        if total_mass is not None:
            raise ValueError("provide total mass or explicit vertex masses, not both")
        masses = np.asarray(explicit_vertex_masses, dtype=np.float64).reshape(-1)
        if len(masses) != vertex_count:
            raise ValueError("explicit vertex mass count does not match the mesh")
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("explicit vertex masses must be finite and positive")
        return masses.copy(), "explicit_per_vertex"
    if total_mass is None:
        raise ValueError("total mass is required without explicit vertex masses")
    return (
        lumped_vertex_masses(elements, rest_volumes, total_mass, vertex_count),
        "volume_lumped_total_mass",
    )


def arap_constraints_and_gradients_numpy(
    positions: np.ndarray,
    elements: np.ndarray,
    rest_inverse: np.ndarray,
    *,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference ARAP constraint/gradient used for finite-difference tests."""

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    elements = np.asarray(elements, dtype=np.int64).reshape(-1, 4)
    rest_inverse = np.asarray(rest_inverse, dtype=np.float64).reshape(-1, 3, 3)
    tets = positions[elements]
    ds = np.stack(
        (tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]),
        axis=-1,
    )
    deformation = ds @ rest_inverse
    left, singular, right_t = np.linalg.svd(deformation)
    orientation = np.linalg.det(left @ right_t)
    reflected = orientation < 0.0
    left[reflected, :, -1] *= -1.0
    singular[reflected, -1] *= -1.0
    delta_sigma = singular - 1.0
    constraints = np.linalg.norm(delta_sigma, axis=1)
    normalized = np.zeros_like(delta_sigma)
    active = constraints > epsilon
    normalized[active] = delta_sigma[active] / constraints[active, None]
    grad_f = (left * normalized[:, None, :]) @ right_t
    grad_ds = grad_f @ np.swapaxes(rest_inverse, 1, 2)
    gradients = np.empty((len(elements), 4, 3), dtype=np.float64)
    gradients[:, 1] = grad_ds[:, :, 0]
    gradients[:, 2] = grad_ds[:, :, 1]
    gradients[:, 3] = grad_ds[:, :, 2]
    gradients[:, 0] = -np.sum(gradients[:, 1:], axis=1)
    gradients[~active] = 0.0
    return constraints, gradients


def greedy_unsmoothed_aggregation(matrix: object, threshold: float) -> np.ndarray:
    """Aggregate a symmetric CSR matrix with the paper's strength threshold."""

    from scipy import sparse

    matrix = sparse.csr_matrix(matrix)
    if matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError("AMG matrix must be non-empty and square")
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    if np.any(diagonal <= 0.0):
        raise ValueError("AMG matrix diagonal must be positive")
    assigned = np.full(matrix.shape[0], -1, dtype=np.int32)
    aggregate = 0
    for row in range(matrix.shape[0]):
        if assigned[row] >= 0:
            continue
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        columns = matrix.indices[start:stop]
        values = np.abs(matrix.data[start:stop])
        offdiag = columns != row
        columns = columns[offdiag]
        values = values[offdiag]
        strong = values >= threshold * np.sqrt(diagonal[row] * diagonal[columns])
        assigned[row] = aggregate
        available = assigned[columns] < 0
        strong_available = available & strong
        selected: list[int] = []
        if np.any(strong_available):
            candidates = columns[strong_available]
            strengths = values[strong_available]
            selected.extend(
                candidates[np.argsort(strengths)[::-1][:7]].tolist()
            )
        # Compliance contributes a large diagonal to the dual matrix.  A
        # purely normalized-strength test can therefore classify every real
        # tet neighbour as weak and leave a 9k-constraint "coarse" level at
        # 7k rows.  Preserve strength priority, then fill the remaining UA
        # aggregate from actual nonzero graph neighbours.  This is a
        # topological fallback, not an algebraic coupling that did not exist.
        remaining_capacity = 7 - len(selected)
        if remaining_capacity > 0:
            weak_available = available & ~strong
            if selected:
                weak_available &= ~np.isin(columns, np.asarray(selected))
            if np.any(weak_available):
                candidates = columns[weak_available]
                strengths = values[weak_available]
                selected.extend(
                    candidates[
                        np.argsort(strengths)[::-1][:remaining_capacity]
                    ].tolist()
                )
        if selected:
            assigned[np.asarray(selected, dtype=np.int64)] = aggregate
        aggregate += 1
    return assigned


def tetrahedralize_proxy(
    path: Path,
    config: MGPBDTetConfig,
    *,
    enclosure_path: Path | None = None,
    volume_path: Path | None = None,
    volume_manifest_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build a well-conditioned volumetric cage around the accepted asset.

    Direct TetGen of the decimated proxy produces boundary slivers as small as
    2e-11 m^3; a sub-millimetre table correction then inverts them.  Use a
    uniform p=4 superellipsoid cage sampled by an icosphere.  Its boundary is
    fitted one contact-particle radius inside the visual surface, because the
    collision projection adds that radius back.  Fitting the bare tet cage to
    the visual and then adding the contact skin made the physical toy several
    millimetres too large for a fully open SO-101 jaw.
    """

    if volume_path is not None or volume_manifest_path is not None:
        if volume_path is None or volume_manifest_path is None:
            raise ValueError("dense volume and manifest paths must be provided together")
        return load_precomputed_tet_mesh(
            volume_path,
            volume_manifest_path,
            expected_contact_radius_m=config.particle_radius_m,
        )

    import trimesh

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MGPBD proxy is missing: {path}")
    surface = trimesh.load(path, process=False, force="mesh")
    if not surface.is_watertight:
        raise ValueError("MGPBD proxy must be watertight")
    started = time.perf_counter()
    enclosure = surface
    if enclosure_path is not None:
        enclosure_path = Path(enclosure_path).resolve()
        if not enclosure_path.is_file():
            raise FileNotFoundError(f"MGPBD enclosure asset is missing: {enclosure_path}")
        enclosure = trimesh.load(enclosure_path, process=False, force="mesh")
    proxy_vertices = np.asarray(surface.vertices, dtype=np.float64)
    enclosure_vertices = np.asarray(enclosure.vertices, dtype=np.float64)
    center = 0.5 * (proxy_vertices.min(axis=0) + proxy_vertices.max(axis=0))
    radii = 0.5 * (proxy_vertices.max(axis=0) - proxy_vertices.min(axis=0))
    if np.any(radii <= 0.0):
        raise ValueError("MGPBD proxy bounds must have positive extent")
    power = 4.0
    normalized_enclosure = (enclosure_vertices - center) / radii
    enclosure_norm = np.sum(np.abs(normalized_enclosure) ** power, axis=1) ** (
        1.0 / power
    )
    radial_distance_m = np.linalg.norm(enclosure_vertices - center, axis=1)
    contact_fit_scale = float(
        np.max(
            enclosure_norm
            * np.maximum(
                0.0,
                1.0
                - config.particle_radius_m
                / np.maximum(radial_distance_m, 1.0e-9),
            )
        )
    )
    outward_margin = 1.01
    scale = contact_fit_scale * outward_margin
    radial_visual_gap_m = np.maximum(
        0.0,
        1.0 - scale / np.maximum(enclosure_norm, 1.0e-9),
    ) * radial_distance_m
    unit = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    directions = np.asarray(unit.vertices, dtype=np.float64)
    lp_norm = np.sum(np.abs(directions) ** power, axis=1) ** (1.0 / power)
    boundary = center + radii * scale * (directions / lp_norm[:, None])
    nodes = np.vstack((boundary, center)).astype(np.float32)
    center_index = len(nodes) - 1
    elements = np.column_stack(
        (
            np.full(len(unit.faces), center_index, dtype=np.int32),
            np.asarray(unit.faces, dtype=np.int32),
        )
    )
    _inverse, volumes = tetrahedral_rest_data(nodes, elements)
    conditions = np.linalg.cond(np.linalg.inv(_inverse))
    diagnostics = {
        "proxy_path": str(path),
        "enclosure_path": str(enclosure_path or path),
        "method": (
            "uniform_icosphere_p4_superellipsoid_contact_skin_fitted_"
            "center_star_tets"
        ),
        "superellipsoid_power": power,
        "outward_margin": outward_margin,
        "particle_contact_radius_m": config.particle_radius_m,
        "contact_fit_scale_before_margin": contact_fit_scale,
        "fitted_scale": scale,
        "maximum_radial_visual_gap_m": float(np.max(radial_visual_gap_m)),
        "contact_skin_encloses_visual": bool(
            np.max(radial_visual_gap_m)
            <= config.particle_radius_m + 1.0e-7
        ),
        "surface_vertices": int(len(surface.vertices)),
        "surface_faces": int(len(surface.faces)),
        "physics_vertices": int(len(nodes)),
        "physics_tetrahedra": int(len(elements)),
        "rest_volume_m3": float(np.sum(volumes)),
        "minimum_tet_volume_m3": float(np.min(volumes)),
        "maximum_rest_matrix_condition": float(np.max(conditions)),
        "tetrahedralize_ms": (time.perf_counter() - started) * 1000.0,
    }
    return nodes, elements, diagnostics


class _TorchAMGLevel:
    def __init__(self, matrix: object, prolongation: object | None, device: object, dtype: object):
        import torch
        from scipy import sparse

        coo = sparse.coo_matrix(matrix)
        indices = torch.as_tensor(
            np.vstack((coo.row, coo.col)), dtype=torch.long, device=device
        )
        values = torch.as_tensor(coo.data, dtype=dtype, device=device)
        self.matrix = torch.sparse_coo_tensor(
            indices, values, coo.shape, dtype=dtype, device=device
        ).coalesce()
        self.diagonal = torch.as_tensor(
            sparse.csr_matrix(matrix).diagonal(), dtype=dtype, device=device
        )
        self.prolongation = None
        if prolongation is not None:
            p = sparse.coo_matrix(prolongation)
            p_indices = torch.as_tensor(
                np.vstack((p.row, p.col)), dtype=torch.long, device=device
            )
            p_values = torch.as_tensor(p.data, dtype=dtype, device=device)
            self.prolongation = torch.sparse_coo_tensor(
                p_indices, p_values, p.shape, dtype=dtype, device=device
            ).coalesce()

    def multiply(self, vector: object):
        import torch

        return torch.sparse.mm(self.matrix, vector[:, None])[:, 0]

    @classmethod
    def from_torch(cls, matrix: object, prolongation: object | None):
        import torch

        instance = cls.__new__(cls)
        instance.matrix = matrix.coalesce()
        indices = instance.matrix.indices()
        diagonal_mask = indices[0] == indices[1]
        diagonal = torch.zeros(
            instance.matrix.shape[0],
            dtype=instance.matrix.dtype,
            device=instance.matrix.device,
        )
        diagonal.index_add_(
            0, indices[0, diagonal_mask], instance.matrix.values()[diagonal_mask]
        )
        instance.diagonal = diagonal
        instance.prolongation = (
            prolongation.coalesce() if prolongation is not None else None
        )
        return instance


class VolumetricMGPBDProjector:
    """One-ARAP-constraint-per-tet dual UA-AMG/MGPCG projector."""

    def __init__(
        self,
        rest_positions: object,
        elements: np.ndarray,
        *,
        total_mass_kg: float | None = None,
        vertex_masses: np.ndarray | None = None,
        config: MGPBDTetConfig | None = None,
    ):
        import torch

        self.config = config or MGPBDTetConfig()
        self.config.validate()
        self.rest_positions = rest_positions.detach().reshape(-1, 3).clone()
        self.device = self.rest_positions.device
        self.dtype = self.rest_positions.dtype
        self.elements_np = np.asarray(elements, dtype=np.int64).reshape(-1, 4)
        rest_np = self.rest_positions.detach().cpu().numpy()
        rest_inverse_np, rest_volumes = tetrahedral_rest_data(rest_np, self.elements_np)
        rest_tets = rest_np[self.elements_np].astype(np.float64)
        rest_signed_six = np.einsum(
            "ij,ij->i",
            np.cross(
                rest_tets[:, 1] - rest_tets[:, 0],
                rest_tets[:, 2] - rest_tets[:, 0],
            ),
            rest_tets[:, 3] - rest_tets[:, 0],
        )
        if np.any(rest_signed_six <= 0.0):
            raise ValueError(
                "MGPBD projector requires positively oriented rest tetrahedra"
            )
        masses, mass_model = resolve_vertex_masses(
            self.elements_np,
            rest_volumes,
            len(rest_np),
            total_mass=total_mass_kg,
            explicit_vertex_masses=vertex_masses,
        )
        self.mass_model = mass_model
        self.elements = torch.as_tensor(
            self.elements_np, dtype=torch.long, device=self.device
        )
        self.rest_inverse = torch.as_tensor(
            rest_inverse_np, dtype=self.dtype, device=self.device
        )
        self.rest_signed_six = torch.as_tensor(
            rest_signed_six, dtype=self.dtype, device=self.device
        )
        self.inverse_mass_np = 1.0 / masses
        self.inverse_mass = torch.as_tensor(
            self.inverse_mass_np, dtype=self.dtype, device=self.device
        )
        self.alpha_tilde_np = self.config.alpha_tilde(rest_volumes)
        self.alpha_tilde = torch.as_tensor(
            self.alpha_tilde_np, dtype=self.dtype, device=self.device
        )
        self._prolongations: list[object] = []
        self._aggregates_np: list[np.ndarray] = []
        self._aggregate_weights_np: list[np.ndarray] = []
        self._levels: list[_TorchAMGLevel] = []
        self._coarsest_cholesky = None
        self._runtime_smoother_weight = self.config.smoother_weight
        self._hierarchy_structure_frame = -1
        self._hierarchy_structure_ready = False
        self._hierarchy_builder = "not_built"
        self._frame = 0
        self._structure_builds = 0
        # P may be reused between frames, but every nonlinear outer iteration
        # must solve against the current J M^-1 J^T + alpha system.  Keep an
        # explicit RAP counter so conformance evidence can prove that the
        # hierarchy values, rather than only its aggregate structure, were
        # refreshed for the matrix being solved.
        self._rap_numeric_refreshes = 0
        (
            contribution_rows,
            contribution_columns,
            contribution_vertices,
            contribution_row_local,
            contribution_column_local,
        ) = self._dual_contribution_pattern(self.elements_np, len(rest_np))
        self._topology_contribution_rows_np = contribution_rows
        self._topology_contribution_columns_np = contribution_columns
        self._contribution_rows = torch.as_tensor(
            contribution_rows, dtype=torch.long, device=self.device
        )
        self._contribution_columns = torch.as_tensor(
            contribution_columns, dtype=torch.long, device=self.device
        )
        self._contribution_vertices = torch.as_tensor(
            contribution_vertices, dtype=torch.long, device=self.device
        )
        self._contribution_row_local = torch.as_tensor(
            contribution_row_local, dtype=torch.long, device=self.device
        )
        self._contribution_column_local = torch.as_tensor(
            contribution_column_local, dtype=torch.long, device=self.device
        )
        self.last_metrics: dict[str, object] = {
            "available": True,
            "projected_frames": 0,
            "constraint_kind": "one_ARAP_singular_value_norm_per_tetrahedron",
        }
        # Diagnostic checkpoint for fail-closed constrained solves.  A
        # projector call is transactional at the frame boundary, but every
        # accepted nonlinear outer state is still useful numerical evidence.
        # Keep it out of the JSON receipt and expose a detached copy instead.
        self._last_accepted_outer_positions = self.rest_positions.detach().clone()
        self._last_accepted_outer_iteration = 0

    @property
    def last_accepted_outer_positions(self):
        """Return the latest atomically accepted nonlinear-outer state."""

        return self._last_accepted_outer_positions.detach().clone()

    @property
    def last_accepted_outer_iteration(self) -> int:
        return int(self._last_accepted_outer_iteration)

    @staticmethod
    def _dual_contribution_pattern(
        elements: np.ndarray, vertex_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Precompute every shared-vertex contribution to ``J M^-1 J^T``."""

        incident: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
        for tet_index, tet in enumerate(elements.tolist()):
            for local_index, vertex in enumerate(tet):
                incident[vertex].append((tet_index, local_index))
        rows: list[int] = []
        columns: list[int] = []
        vertices: list[int] = []
        row_local: list[int] = []
        column_local: list[int] = []
        for vertex, entries in enumerate(incident):
            for row, row_slot in entries:
                for column, column_slot in entries:
                    rows.append(row)
                    columns.append(column)
                    vertices.append(vertex)
                    row_local.append(row_slot)
                    column_local.append(column_slot)
        return tuple(
            np.asarray(values, dtype=np.int64)
            for values in (rows, columns, vertices, row_local, column_local)
        )

    def constraints_and_gradients(self, positions: object):
        import torch

        indices = torch.arange(
            len(self.elements_np), dtype=torch.long, device=self.device
        )
        return self._constraints_and_gradients_for_tets(positions, indices)

    def _constraints_and_gradients_for_tets(
        self, positions: object, tet_indices: object
    ):
        """Evaluate ARAP rows for a selected set of tetrahedra."""

        import torch

        selected_elements = self.elements[tet_indices]
        selected_rest_inverse = self.rest_inverse[tet_indices]
        tets = positions[selected_elements]
        ds = torch.stack(
            (tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]),
            dim=-1,
        )
        deformation = ds @ selected_rest_inverse
        left, singular, right_t = torch.linalg.svd(deformation)
        # Proper-rotation ARAP must not treat a reflection as strain-free.
        # Torch returns nonnegative singular values, whereas the Taichi SVD
        # used by the public MGPBD soft-body code keeps U/V as rotations and
        # carries det(F)'s sign in the last singular value.
        orientation = torch.linalg.det(left @ right_t)
        reflected = orientation < 0.0
        left = left.clone()
        singular = singular.clone()
        left[reflected, :, -1] *= -1.0
        singular[reflected, -1] *= -1.0
        delta_sigma = singular - 1.0
        constraints = torch.linalg.norm(delta_sigma, dim=1)
        active = constraints > self.config.constraint_epsilon
        normalized = torch.where(
            active[:, None],
            delta_sigma / constraints.clamp_min(self.config.constraint_epsilon)[:, None],
            torch.zeros_like(delta_sigma),
        )
        grad_f = (left * normalized[:, None, :]) @ right_t
        grad_ds = grad_f @ selected_rest_inverse.transpose(1, 2)
        gradients = torch.empty(
            (len(tet_indices), 4, 3), dtype=self.dtype, device=self.device
        )
        gradients[:, 1] = grad_ds[:, :, 0]
        gradients[:, 2] = grad_ds[:, :, 1]
        gradients[:, 3] = grad_ds[:, :, 2]
        gradients[:, 0] = -torch.sum(gradients[:, 1:], dim=1)
        gradients = torch.where(active[:, None, None], gradients, torch.zeros_like(gradients))
        return constraints, gradients, active

    def signed_volume_ratios(self, positions: object):
        """Return current signed six-volume divided by positive rest value."""

        import torch

        tets = positions[self.elements]
        signed_six = torch.sum(
            torch.cross(
                tets[:, 1] - tets[:, 0],
                tets[:, 2] - tets[:, 0],
                dim=1,
            )
            * (tets[:, 3] - tets[:, 0]),
            dim=1,
        )
        return signed_six / self.rest_signed_six

    def signed_volume_ratios_and_gradients(self, positions: object):
        """Return signed-volume ratios and exact local position gradients."""

        import torch

        indices = torch.arange(
            len(self.elements_np), dtype=torch.long, device=self.device
        )
        return self._signed_volume_ratios_and_gradients_for_tets(
            positions, indices
        )

    def _signed_volume_ratios_and_gradients_for_tets(
        self, positions: object, tet_indices: object
    ):
        """Evaluate signed-volume rows for selected tetrahedra."""

        import torch

        selected_elements = self.elements[tet_indices]
        tets = positions[selected_elements]
        edge_1 = tets[:, 1] - tets[:, 0]
        edge_2 = tets[:, 2] - tets[:, 0]
        edge_3 = tets[:, 3] - tets[:, 0]
        gradients = torch.empty(
            (len(tet_indices), 4, 3),
            dtype=self.dtype,
            device=self.device,
        )
        denominator = self.rest_signed_six[tet_indices, None]
        gradients[:, 1] = torch.cross(edge_2, edge_3, dim=1) / denominator
        gradients[:, 2] = torch.cross(edge_3, edge_1, dim=1) / denominator
        gradients[:, 3] = torch.cross(edge_1, edge_2, dim=1) / denominator
        gradients[:, 0] = -torch.sum(gradients[:, 1:], dim=1)
        ratios = torch.sum(
            torch.cross(edge_1, edge_2, dim=1) * edge_3, dim=1
        ) / self.rest_signed_six[tet_indices]
        return ratios, gradients

    def _local_rows_directional(
        self,
        rows: object,
        direction: object,
        tet_indices: object | None = None,
    ):
        """Apply one local four-vertex row per tetrahedron to a direction."""

        selected_elements = (
            self.elements if tet_indices is None else self.elements[tet_indices]
        )
        return (rows * direction[selected_elements]).sum(dim=(1, 2))

    def _local_rows_transpose(
        self,
        rows: object,
        coefficients: object,
        tet_indices: object | None = None,
    ):
        """Scatter local constraint rows transposed into vertex-vector space."""

        import torch

        selected_elements = self.elements if tet_indices is None else self.elements[tet_indices]
        result = torch.zeros_like(self.rest_positions)
        weighted = coefficients[:, None, None] * rows
        for local_index in range(4):
            result.index_add_(
                0, selected_elements[:, local_index], weighted[:, local_index]
            )
        return result

    def _local_rows_mass_gram(self, rows: object, tet_indices: object):
        """Return ``B M^-1 B^T`` without expanding local rows over vertices.

        Every safety row has support on exactly four tetrahedron vertices.  A
        pair contributes only when two local slots reference the same global
        vertex, so sixteen ``K x K`` masked products replace the former
        ``K x vertex_count x 3`` dense working set.  The Gram matrix remains in
        FP64, matching the Schur factorization path.
        """

        import torch

        count = len(tet_indices)
        gram64 = torch.zeros(
            (count, count), dtype=torch.float64, device=self.device
        )
        if count == 0:
            return gram64
        vertices = self.elements[tet_indices]
        rows64 = rows.to(torch.float64)
        for left_local in range(4):
            left_vertices = vertices[:, left_local]
            left_inverse_mass64 = self.inverse_mass[left_vertices].to(
                torch.float64
            )
            left_rows64 = rows64[:, left_local]
            for right_local in range(4):
                shared_vertex = left_vertices[:, None] == vertices[
                    None, :, right_local
                ]
                dot_products64 = (
                    left_rows64 @ rows64[:, right_local].transpose(0, 1)
                )
                gram64 += torch.where(
                    shared_vertex,
                    left_inverse_mass64[:, None] * dot_products64,
                    torch.zeros_like(dot_products64),
                )
        return gram64

    def _assemble_cpu_matrix(self, gradients: object):
        from scipy import sparse

        grad = gradients.detach().cpu().numpy().astype(np.float64, copy=False)
        tet_count = len(self.elements_np)
        rows = np.repeat(np.arange(tet_count, dtype=np.int64), 12)
        columns = (
            self.elements_np[:, :, None] * 3 + np.arange(3, dtype=np.int64)
        ).reshape(-1)
        jacobian = sparse.coo_matrix(
            (grad.reshape(-1), (rows, columns)),
            shape=(tet_count, 3 * len(self.rest_positions)),
        ).tocsr()
        inverse_mass_xyz = np.repeat(self.inverse_mass_np, 3)
        weighted = jacobian.multiply(inverse_mass_xyz[None, :])
        matrix = weighted @ jacobian.T
        matrix = matrix + sparse.diags(self.alpha_tilde_np, format="csr")
        return sparse.csr_matrix(matrix, dtype=np.float64)

    def _build_prolongations(self, fine_matrix: object) -> None:
        from scipy import sparse

        matrix = sparse.csr_matrix(fine_matrix)
        use_topology_hierarchy = self.config.amg_hierarchy_mode == "topology_ua" or (
            self.config.amg_hierarchy_mode == "auto" and len(self.elements_np) >= 1_000
        )
        if use_topology_hierarchy:
            # Build dense-volume aggregates from immutable tet topology, not
            # the current ARAP Jacobian.  At near rest most ARAP gradients are
            # exactly zero, so the dynamic dual matrix contains only alpha on
            # thousands of rows and cannot reveal that neighbouring tets share
            # vertices.  The prolongation structure is geometric and fixed;
            # only RAP coefficients are refreshed from the dynamic matrix.
            topology = sparse.coo_matrix(
                (
                    np.ones(
                        len(self._topology_contribution_rows_np),
                        dtype=np.float64,
                    ),
                    (
                        self._topology_contribution_rows_np,
                        self._topology_contribution_columns_np,
                    ),
                ),
                shape=matrix.shape,
            ).tocsr()
            topology.data.fill(1.0)
            prolongations: list[object] = []
            aggregates: list[np.ndarray] = []
            aggregate_weights: list[np.ndarray] = []
            for _ in range(self.config.amg_max_levels - 1):
                if topology.shape[0] <= self.config.amg_coarsest_size:
                    break
                aggregate = greedy_unsmoothed_aggregation(topology, 0.0)
                coarse_count = int(aggregate.max()) + 1
                if coarse_count >= topology.shape[0]:
                    break
                counts = np.bincount(aggregate, minlength=coarse_count)
                weights = 1.0 / np.sqrt(counts[aggregate])
                prolongation = sparse.coo_matrix(
                    (weights, (np.arange(len(aggregate)), aggregate)),
                    shape=(len(aggregate), coarse_count),
                ).tocsr()
                prolongations.append(prolongation)
                aggregates.append(aggregate.copy())
                aggregate_weights.append(weights.copy())
                topology = (prolongation.T @ topology @ prolongation).tocsr()
                topology.data.fill(1.0)
                topology.eliminate_zeros()
            self._prolongations = prolongations
            self._aggregates_np = aggregates
            self._aggregate_weights_np = aggregate_weights
            self._hierarchy_structure_frame = self._frame
            self._hierarchy_structure_ready = True
            self._structure_builds += 1
            self._hierarchy_builder = "static_tet_topology_greedy_UA"
            return

        # Reproduce the authors' public ``build_Ps.py`` recipe when PyAMG is
        # present.  The exact hierarchy also depends on the recorded
        # PyAMG/SciPy versions, so this is a clean-room recipe match rather
        # than an official binary identity claim.  With ``smooth=None`` every
        # tentative-UA row has one aggregate entry, which also lets the
        # Radeon runtime rebuild RAP by index reduction without a
        # sparse-sparse dependency.
        public_ua_error: Exception | None = None
        try:
            import pyamg

            multilevel = pyamg.smoothed_aggregation_solver(
                matrix,
                max_coarse=self.config.amg_coarsest_size,
                max_levels=self.config.amg_max_levels,
                smooth=None,
                improve_candidates=None,
                symmetry="symmetric",
            )
            official_prolongations: list[object] = []
            official_aggregates: list[np.ndarray] = []
            official_weights: list[np.ndarray] = []
            compatible = True
            for level in multilevel.levels[:-1]:
                prolongation = sparse.csr_matrix(level.P)
                row_counts = np.diff(prolongation.indptr)
                if np.any(row_counts != 1):
                    compatible = False
                    break
                official_prolongations.append(prolongation)
                official_aggregates.append(prolongation.indices.copy().astype(np.int32))
                official_weights.append(prolongation.data.copy().astype(np.float64))
            if compatible and official_prolongations:
                self._prolongations = official_prolongations
                self._aggregates_np = official_aggregates
                self._aggregate_weights_np = official_weights
                self._hierarchy_structure_frame = self._frame
                self._hierarchy_structure_ready = True
                self._structure_builds += 1
                self._hierarchy_builder = (
                    "PyAMG_plain_UA_smooth_none_clean_room"
                )
                return
            public_ua_error = RuntimeError(
                "PyAMG public UA did not produce one-entry tentative prolongations"
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            public_ua_error = exc
        if self.config.amg_hierarchy_mode == "matrix_ua":
            raise RuntimeError("explicit matrix-UA hierarchy construction failed") from public_ua_error

        prolongations: list[object] = []
        aggregates: list[np.ndarray] = []
        aggregate_weights: list[np.ndarray] = []
        for _ in range(self.config.amg_max_levels - 1):
            if matrix.shape[0] <= self.config.amg_coarsest_size:
                break
            aggregate = greedy_unsmoothed_aggregation(
                matrix, self.config.amg_strength_threshold
            )
            coarse_count = int(aggregate.max()) + 1
            if coarse_count >= matrix.shape[0]:
                break
            counts = np.bincount(aggregate, minlength=coarse_count)
            weights = 1.0 / np.sqrt(counts[aggregate])
            prolongation = sparse.coo_matrix(
                (weights, (np.arange(len(aggregate)), aggregate)),
                shape=(len(aggregate), coarse_count),
            ).tocsr()
            prolongations.append(prolongation)
            aggregates.append(aggregate.copy())
            aggregate_weights.append(weights.copy())
            matrix = (prolongation.T @ matrix @ prolongation).tocsr()
        self._prolongations = prolongations
        self._aggregates_np = aggregates
        self._aggregate_weights_np = aggregate_weights
        self._hierarchy_structure_frame = self._frame
        self._hierarchy_structure_ready = True
        self._structure_builds += 1
        self._hierarchy_builder = (
            "internal_strength_then_structural_greedy_UA_"
            f"threshold_{self.config.amg_strength_threshold:g}"
        )

    def _refresh_levels(self, fine_matrix: object, *, rebuild_structure: bool) -> None:
        import torch
        from scipy import sparse

        if rebuild_structure:
            self._build_prolongations(fine_matrix)
        matrices = [sparse.csr_matrix(fine_matrix)]
        for prolongation in self._prolongations:
            matrices.append(
                (prolongation.T @ matrices[-1] @ prolongation).tocsr()
            )
        if self.config.smoother_weight_mode == "fine_spectral_radius":
            from pyamg.relaxation.smoothing import rho_D_inv_A

            spectral_radius = float(rho_D_inv_A(matrices[0]))
            if not np.isfinite(spectral_radius) or spectral_radius <= 0.0:
                raise RuntimeError("invalid fine-level Jacobi spectral radius")
            self._runtime_smoother_weight = 1.0 / spectral_radius
        else:
            self._runtime_smoother_weight = self.config.smoother_weight
        self._levels = []
        for index, matrix in enumerate(matrices):
            prolongation = (
                self._prolongations[index]
                if index < len(self._prolongations)
                else None
            )
            self._levels.append(
                _TorchAMGLevel(matrix, prolongation, self.device, self.dtype)
            )
        coarsest_np = matrices[-1].toarray()
        coarsest_np = 0.5 * (coarsest_np + coarsest_np.T)
        dense = torch.as_tensor(coarsest_np, dtype=self.dtype, device=self.device)
        jitter = max(float(np.max(np.diag(coarsest_np))) * 1.0e-7, 1.0e-9)
        try:
            self._coarsest_cholesky = torch.linalg.cholesky(
                dense
                + torch.eye(len(dense), dtype=self.dtype, device=self.device) * jitter
            )
        except RuntimeError:
            self._coarsest_cholesky = None
        self._rap_numeric_refreshes += 1

    def _assemble_torch_system(self, gradients: object, diagonal: object):
        import torch

        row_gradients = gradients[
            self._contribution_rows, self._contribution_row_local
        ]
        column_gradients = gradients[
            self._contribution_columns, self._contribution_column_local
        ]
        values = (
            self.inverse_mass[self._contribution_vertices]
            * torch.sum(row_gradients * column_gradients, dim=1)
        )
        tet_count = len(self.elements_np)
        diagonal_indices = torch.arange(
            tet_count, dtype=torch.long, device=self.device
        )
        rows = torch.cat((self._contribution_rows, diagonal_indices))
        columns = torch.cat((self._contribution_columns, diagonal_indices))
        values = torch.cat((values, self.alpha_tilde))
        scaling = (
            torch.rsqrt(diagonal.clamp_min(1.0e-20))
            if self.config.symmetric_diagonal_equilibration
            else torch.ones_like(diagonal)
        )
        values = values * scaling[rows] * scaling[columns]
        return torch.sparse_coo_tensor(
            torch.stack((rows, columns)),
            values,
            (tet_count, tet_count),
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def _refresh_levels_gpu(self, fine_matrix: object) -> None:
        """Recompute all RAP values on Radeon while reusing lazy UA groups."""

        import torch

        matrices = [fine_matrix.coalesce()]
        prolongations: list[object] = []
        current = matrices[0]
        for aggregate_np, weights_np in zip(
            self._aggregates_np, self._aggregate_weights_np, strict=True
        ):
            aggregate = torch.as_tensor(
                aggregate_np, dtype=torch.long, device=self.device
            )
            weights = torch.as_tensor(
                weights_np, dtype=self.dtype, device=self.device
            )
            fine_count = len(aggregate_np)
            coarse_count = int(np.max(aggregate_np)) + 1
            fine_rows = torch.arange(
                fine_count, dtype=torch.long, device=self.device
            )
            prolongation = torch.sparse_coo_tensor(
                torch.stack((fine_rows, aggregate)),
                weights,
                (fine_count, coarse_count),
                dtype=self.dtype,
                device=self.device,
            ).coalesce()
            prolongations.append(prolongation)
            indices = current.indices()
            coarse_rows = aggregate[indices[0]]
            coarse_columns = aggregate[indices[1]]
            coarse_values = (
                current.values()
                * weights[indices[0]]
                * weights[indices[1]]
            )
            current = torch.sparse_coo_tensor(
                torch.stack((coarse_rows, coarse_columns)),
                coarse_values,
                (coarse_count, coarse_count),
                dtype=self.dtype,
                device=self.device,
            ).coalesce()
            matrices.append(current)
        self._levels = [
            _TorchAMGLevel.from_torch(
                matrix,
                prolongations[index] if index < len(prolongations) else None,
            )
            for index, matrix in enumerate(matrices)
        ]
        coarsest = matrices[-1].to_dense()
        coarsest = 0.5 * (coarsest + coarsest.T)
        jitter = torch.max(torch.diagonal(coarsest)) * 1.0e-7 + 1.0e-9
        try:
            self._coarsest_cholesky = torch.linalg.cholesky(
                coarsest
                + torch.eye(
                    len(coarsest), dtype=self.dtype, device=self.device
                )
                * jitter
            )
        except RuntimeError:
            self._coarsest_cholesky = None
        self._rap_numeric_refreshes += 1

    def _operator(self, vector: object, gradients: object):
        import torch

        impulses = torch.zeros_like(self.rest_positions)
        weighted = vector[:, None, None] * gradients
        for local_index in range(4):
            impulses.index_add_(
                0, self.elements[:, local_index], weighted[:, local_index]
            )
        impulses *= self.inverse_mass[:, None]
        gathered = impulses[self.elements]
        return torch.sum(gathered * gradients, dim=(1, 2)) + self.alpha_tilde * vector

    def _smooth(self, level_index: int, rhs: object, solution: object):
        level = self._levels[level_index]
        for _ in range(self.config.smoother_iterations):
            residual = rhs - level.multiply(solution)
            solution = (
                solution
                + self._runtime_smoother_weight * residual / level.diagonal
            )
        return solution

    def _v_cycle(self, level_index: int, rhs: object):
        import torch

        level = self._levels[level_index]
        if level_index == len(self._levels) - 1 or level.prolongation is None:
            if self._coarsest_cholesky is None:
                return rhs / level.diagonal
            return torch.cholesky_solve(rhs[:, None], self._coarsest_cholesky)[:, 0]
        solution = torch.zeros_like(rhs)
        solution = self._smooth(level_index, rhs, solution)
        residual = rhs - level.multiply(solution)
        restricted = torch.sparse.mm(
            level.prolongation.transpose(0, 1), residual[:, None]
        )[:, 0]
        coarse = self._v_cycle(level_index + 1, restricted)
        solution = solution + torch.sparse.mm(
            level.prolongation, coarse[:, None]
        )[:, 0]
        return self._smooth(level_index, rhs, solution)

    def _precondition(self, residual: object, diagonal: object):
        if not self._levels:
            return residual / diagonal
        return self._v_cycle(0, residual)

    def _pcg(self, rhs: object, gradients: object, diagonal: object):
        import torch

        # TetGen's boundary-conforming mesh can span many element scales.  A
        # symmetric Jacobi equilibration keeps the mathematically identical
        # dual system representable in FP32 on Radeon:
        #   (S A S) y = S b,  delta_lambda = S y.
        # This is a variable scaling, not an approximate local solve.
        scaling = (
            torch.rsqrt(diagonal.clamp_min(1.0e-20))
            if self.config.symmetric_diagonal_equilibration
            else torch.ones_like(diagonal)
        )
        scaled_diagonal = diagonal * scaling * scaling
        scaled_rhs = scaling * rhs
        solution = torch.zeros_like(scaled_rhs)
        residual = scaled_rhs.clone()
        initial_norm = torch.linalg.norm(residual)
        if float(initial_norm) <= 1.0e-12:
            return solution, 0, 0.0, 0.0, 0.0, 0
        rhs_norm = torch.linalg.norm(rhs).clamp_min(1.0e-30)
        preconditioned = self._precondition(residual, scaled_diagonal)
        search = preconditioned.clone()
        rz = torch.dot(residual, preconditioned)
        if not bool(torch.isfinite(rz)) or float(torch.abs(rz)) <= 1.0e-20:
            return solution, 0, 1.0, 1.0, 0.0, 0
        relative = 1.0
        recursive_relative = 1.0
        iterations = 0
        residual_replacements = 0
        for iteration in range(self.config.pcg_iterations):
            # Once the current dual matrix has been assembled for RAP, its
            # level-0 sparse matvec is exactly S A S and avoids four Python
            # index_add launches per Krylov iteration on the Radeon backend.
            operator_search = (
                self._levels[0].multiply(search)
                if self._levels
                else scaling * self._operator(scaling * search, gradients)
            )
            denominator = torch.dot(search, operator_search)
            if not bool(torch.isfinite(denominator)) or float(denominator) <= 1.0e-20:
                break
            step = rz / denominator
            solution = solution + step * search
            residual = residual - step * operator_search
            relative = float(torch.linalg.norm(residual) / initial_norm)
            recursive_relative = relative
            iterations = iteration + 1
            # FP32 recursive residuals drift on the high-stiffness bunny.
            # Periodically replace r with the residual of the physical
            # unscaled operator, and only declare convergence from that true
            # residual.  Restarting the Krylov direction after replacement
            # preserves PCG correctness instead of continuing with a stale
            # conjugacy recurrence.
            if (
                relative <= self.config.relative_residual
                or iterations % 100 == 0
            ):
                delta_lambda_check = scaling * solution
                true_unscaled = rhs - self._operator(
                    delta_lambda_check, gradients
                )
                true_relative_check = float(
                    torch.linalg.norm(true_unscaled) / rhs_norm
                )
                if true_relative_check <= self.config.relative_residual:
                    break
                residual = scaling * true_unscaled
                relative = float(torch.linalg.norm(residual) / initial_norm)
                preconditioned = self._precondition(residual, scaled_diagonal)
                search = preconditioned.clone()
                rz = torch.dot(residual, preconditioned)
                residual_replacements += 1
                if (
                    not bool(torch.isfinite(rz))
                    or float(torch.abs(rz)) <= 1.0e-20
                ):
                    break
                continue
            next_preconditioned = self._precondition(residual, scaled_diagonal)
            next_rz = torch.dot(residual, next_preconditioned)
            if (
                not bool(torch.isfinite(next_rz))
                or float(torch.abs(next_rz)) <= 1.0e-20
                or float(torch.abs(rz)) <= 1.0e-20
            ):
                break
            search = next_preconditioned + (next_rz / rz) * search
            rz = next_rz
        delta_lambda = scaling * solution
        physical_action = self._operator(delta_lambda, gradients)
        true_relative = float(
            torch.linalg.norm(rhs - physical_action) / rhs_norm
        )
        if self._levels:
            level_action = self._levels[0].multiply(solution)
            scaled_physical_action = scaling * physical_action
            operator_denominator = torch.linalg.norm(
                scaled_physical_action
            ).clamp_min(1.0e-30)
            level0_operator_relative_error = float(
                torch.linalg.norm(level_action - scaled_physical_action)
                / operator_denominator
            )
        else:
            level0_operator_relative_error = 0.0
        return (
            delta_lambda,
            iterations,
            recursive_relative,
            true_relative,
            level0_operator_relative_error,
            residual_replacements,
        )

    def _sqp_constrained_direction_unbatched_reference(
        self,
        current: object,
        material_residual: object,
        constraints: object,
        gradients: object,
        base_direction: object,
        base_delta_lambda: object,
        diagonal: object,
    ):
        """Superseded single-cut prototype retained during P0 bring-up."""

        import torch

        config = self.config
        tet_count = len(self.elements_np)
        ratios, volume_gradients = self.signed_volume_ratios_and_gradients(
            current
        )
        raw_positions = current + base_direction
        raw_constraints, raw_gradients, _raw_active = (
            self.constraints_and_gradients(raw_positions)
        )
        raw_ratios = self.signed_volume_ratios(raw_positions)
        restoration = (
            constraints >= config.sqp_strain_activation_threshold
        ) & (raw_constraints > config.sqp_strain_maximum)
        safety_strain_gradients = torch.where(
            restoration[:, None, None], raw_gradients, gradients
        )
        raw_cut_directional = self._local_rows_directional(
            raw_gradients, base_direction
        )
        strain_bounds = config.sqp_strain_fraction_to_boundary * (
            config.sqp_strain_maximum - constraints
        )
        restoration_bounds = (
            config.sqp_strain_maximum
            - raw_constraints
            + raw_cut_directional
        )
        strain_bounds = torch.where(
            restoration, restoration_bounds, strain_bounds
        )
        volume_bounds = config.sqp_volume_fraction_to_boundary * (
            ratios - config.sqp_minimum_signed_volume_ratio
        )
        inverse_mass_tets = self.inverse_mass[self.elements]
        strain_norm = torch.sqrt(
            torch.sum(
                inverse_mass_tets * torch.sum(safety_strain_gradients**2, dim=2),
                dim=1,
            )
        ).clamp_min(1.0e-20)
        volume_norm = torch.sqrt(
            torch.sum(
                inverse_mass_tets * torch.sum(volume_gradients**2, dim=2),
                dim=1,
            )
        ).clamp_min(1.0e-20)
        normalized_strain_gradients = (
            safety_strain_gradients / strain_norm[:, None, None]
        )
        normalized_volume_gradients = (
            -volume_gradients / volume_norm[:, None, None]
        )
        normalized_bounds = torch.cat(
            (strain_bounds / strain_norm, volume_bounds / volume_norm)
        )
        determinant_eligible = (
            ratios <= config.sqp_determinant_activation_ratio
        )
        eligible = torch.cat(
            (
                torch.ones(
                    tet_count, dtype=torch.bool, device=self.device
                ),
                determinant_eligible,
            )
        )

        def all_directional(direction: object):
            return torch.cat(
                (
                    self._local_rows_directional(
                        normalized_strain_gradients, direction
                    ),
                    self._local_rows_directional(
                        normalized_volume_gradients, direction
                    ),
                )
            )

        def active_rows(indices: list[int]):
            index_tensor = torch.as_tensor(
                indices, dtype=torch.long, device=self.device
            )
            is_volume = index_tensor >= tet_count
            tet_indices = torch.remainder(index_tensor, tet_count)
            rows = torch.where(
                is_volume[:, None, None],
                normalized_volume_gradients[tet_indices],
                normalized_strain_gradients[tet_indices],
            )
            return index_tensor, tet_indices, rows

        initial_violation = all_directional(base_direction) - normalized_bounds
        initial_restoration = torch.nonzero(
            restoration
            & (
                initial_violation[:tet_count]
                > config.sqp_primal_tolerance
            ),
            as_tuple=False,
        ).flatten()
        active = [int(index) for index in initial_restoration.detach().cpu()]
        if len(active) > config.sqp_maximum_active_constraints:
            raise RuntimeError("SQP restoration working set exceeds its limit")
        direction = torch.zeros_like(base_direction)
        material_adjustment = torch.zeros_like(base_delta_lambda)
        multipliers64 = torch.empty(
            0, dtype=torch.float64, device=self.device
        )
        coupling_cache: dict[int, object] = {}
        inverse_coupling_cache: dict[int, object] = {}
        auxiliary_residuals: dict[int, float] = {}
        additions = 0
        removals = 0
        converged = False
        schur_minimum_diagonal = float("inf")
        schur_minimum_cholesky_diagonal = float("inf")

        for active_iteration in range(
            1, config.sqp_maximum_active_set_iterations + 1
        ):
            if active:
                _index_tensor, active_tets, local_rows = active_rows(active)
                missing = [
                    index for index in active if index not in coupling_cache
                ]
                for index in missing:
                    _single_index, single_tet, single_row = active_rows([index])
                    weighted_position = self._local_rows_transpose(
                        single_row,
                        torch.ones(
                            1, dtype=self.dtype, device=self.device
                        ),
                        single_tet,
                    )
                    weighted_position *= self.inverse_mass[:, None]
                    coupling_rhs = self._local_rows_directional(
                        gradients, weighted_position
                    )
                    (
                        inverse_coupling,
                        _aux_iterations,
                        _aux_recursive,
                        aux_true,
                        _aux_operator_error,
                        _aux_replacements,
                    ) = self._pcg(coupling_rhs, gradients, diagonal)
                    coupling_cache[index] = coupling_rhs
                    inverse_coupling_cache[index] = inverse_coupling
                    auxiliary_residuals[index] = float(aux_true)
                coupling = torch.stack(
                    [coupling_cache[index] for index in active], dim=1
                )
                inverse_coupling = torch.stack(
                    [inverse_coupling_cache[index] for index in active],
                    dim=1,
                )
                direct_gram64 = self._local_rows_mass_gram(
                    local_rows, active_tets
                )
                gram64 = direct_gram64 - (
                    coupling.to(torch.float64).transpose(0, 1)
                    @ inverse_coupling.to(torch.float64)
                )
                gram64 = 0.5 * (gram64 + gram64.transpose(0, 1))
                schur_minimum_diagonal = min(
                    schur_minimum_diagonal,
                    float(torch.min(torch.diagonal(gram64))),
                )
                rhs64 = (
                    all_directional(base_direction)[
                        torch.as_tensor(
                            active, dtype=torch.long, device=self.device
                        )
                    ]
                    - normalized_bounds[
                        torch.as_tensor(
                            active, dtype=torch.long, device=self.device
                        )
                    ]
                ).to(torch.float64)
                cholesky64, cholesky_info = torch.linalg.cholesky_ex(gram64)
                if bool(torch.any(cholesky_info != 0)):
                    raise RuntimeError(
                        "SQP active Schur complement is not positive definite"
                    )
                schur_minimum_cholesky_diagonal = min(
                    schur_minimum_cholesky_diagonal,
                    float(torch.min(torch.diagonal(cholesky64))),
                )
                multipliers64 = torch.cholesky_solve(
                    rhs64[:, None], cholesky64
                )[:, 0]
                target_material_adjustment = (
                    inverse_coupling
                    @ multipliers64.to(self.dtype)
                )
                safety_impulse = self._local_rows_transpose(
                    local_rows,
                    multipliers64.to(self.dtype),
                    active_tets,
                )
                material_impulse = self._local_rows_transpose(
                    gradients, target_material_adjustment
                )
                target = base_direction + self.inverse_mass[:, None] * (
                    material_impulse - safety_impulse
                )
            else:
                target = base_direction
                target_material_adjustment = torch.zeros_like(
                    base_delta_lambda
                )
                multipliers64 = torch.empty(
                    0, dtype=torch.float64, device=self.device
                )

            search = target - direction
            directional = all_directional(search)
            slack = normalized_bounds - all_directional(direction)
            inactive = eligible.clone()
            if active:
                inactive[
                    torch.as_tensor(
                        active, dtype=torch.long, device=self.device
                    )
                ] = False
            blocking = (
                inactive
                & (directional > config.sqp_primal_tolerance)
                & (
                    slack
                    < directional * (1.0 - config.sqp_primal_tolerance)
                )
            )
            if bool(torch.any(blocking)):
                fractions = torch.where(
                    blocking,
                    torch.clamp(slack / directional.clamp_min(1.0e-30), 0.0, 1.0),
                    torch.full_like(slack, float("inf")),
                )
                active_step, blocking_index = torch.min(fractions, dim=0)
                direction = direction + active_step * search
                material_adjustment = material_adjustment + active_step * (
                    target_material_adjustment - material_adjustment
                )
                if len(active) >= config.sqp_maximum_active_constraints:
                    raise RuntimeError("SQP active set reached its limit")
                active.append(int(blocking_index))
                additions += 1
                continue

            direction = target
            material_adjustment = target_material_adjustment
            negative = torch.nonzero(
                multipliers64 < -config.sqp_dual_tolerance,
                as_tuple=False,
            ).flatten()
            if len(negative):
                local_remove = int(
                    negative[
                        torch.argmin(multipliers64[negative])
                    ]
                )
                del active[local_remove]
                removals += 1
                continue
            multipliers64 = torch.clamp_min(multipliers64, 0.0)
            final_violation = all_directional(direction) - normalized_bounds
            if float(torch.max(final_violation[eligible])) <= (
                config.sqp_primal_tolerance
            ):
                converged = True
                break
        else:
            active_iteration = config.sqp_maximum_active_set_iterations

        if not converged:
            raise RuntimeError("SQP active set did not converge")

        adjusted_delta_lambda = base_delta_lambda + material_adjustment
        linearized_residual_direction = (
            self._local_rows_directional(gradients, direction)
            + self.alpha_tilde * adjusted_delta_lambda
        )
        linearized_material_residual = (
            material_residual + linearized_residual_direction
        )
        merit_slope = float(
            torch.dot(material_residual, linearized_residual_direction)
        )
        if not np.isfinite(merit_slope) or merit_slope >= 0.0:
            raise RuntimeError("SQP constrained direction is not a merit descent")

        final_violation = all_directional(direction) - normalized_bounds
        direction_norm = torch.linalg.norm(base_direction).clamp_min(1.0e-30)
        constrained_norm = torch.linalg.norm(direction).clamp_min(1.0e-30)
        cosine = float(
            torch.sum(base_direction * direction)
            / (direction_norm * constrained_norm)
        )
        active_indices = np.asarray(active, dtype=np.int64)
        active_digest = hashlib.sha256(active_indices.tobytes()).hexdigest()
        metrics = {
            "enabled": True,
            "backend": "Torch_ROCm_MGPCG_Schur_active_set",
            "converged": True,
            "fallback_used": False,
            "active_set_iterations": int(active_iteration),
            "active_constraints": len(active),
            "active_strain_constraints": int(
                np.count_nonzero(active_indices < tet_count)
            ),
            "active_determinant_constraints": int(
                np.count_nonzero(active_indices >= tet_count)
            ),
            "active_set_additions": additions,
            "active_set_removals": removals,
            "active_indices_head": active_indices[:32].tolist(),
            "active_indices_sha256": active_digest,
            "nonlinear_strain_restoration_candidates": int(
                torch.count_nonzero(restoration)
            ),
            "monitored_determinant_constraints": int(
                torch.count_nonzero(determinant_eligible)
            ),
            "initial_maximum_linearized_violation": float(
                torch.max(initial_violation[eligible])
            ),
            "final_maximum_linearized_violation": float(
                torch.max(final_violation[eligible])
            ),
            "minimum_multiplier": (
                float(torch.min(multipliers64)) if len(multipliers64) else 0.0
            ),
            "maximum_auxiliary_true_relative_residual": (
                max(auxiliary_residuals.values())
                if auxiliary_residuals
                else 0.0
            ),
            "auxiliary_linear_solves": len(auxiliary_residuals),
            "schur_minimum_diagonal": (
                schur_minimum_diagonal if active else 0.0
            ),
            "schur_minimum_cholesky_diagonal": (
                schur_minimum_cholesky_diagonal if active else 0.0
            ),
            "linearized_material_residual_l2": float(
                torch.linalg.norm(linearized_material_residual)
            ),
            "merit_slope": merit_slope,
            "raw_full_candidate_arap_maximum": float(
                torch.max(raw_constraints)
            ),
            "raw_full_candidate_minimum_signed_volume_ratio": float(
                torch.min(raw_ratios)
            ),
            "base_to_constrained_direction_cosine": cosine,
            "direction_change_l2": float(
                torch.linalg.norm(direction - base_direction)
            ),
        }
        return direction, adjusted_delta_lambda, metrics

    def _sqp_constrained_direction(
        self,
        current: object,
        material_residual: object,
        constraints: object,
        gradients: object,
        base_direction: object,
        base_delta_lambda: object,
        diagonal: object,
    ):
        """Solve a cut-resolved H-metric safety SQP on the current MGPBD A.

        Current-state strain and determinant rows are always retained.  If the
        full constrained direction is still nonlinearly unsafe, tangent rows
        are generated at the first boundary crossings and the same quadratic
        subproblem is solved again.  Scalar backtracking is reserved for the
        coupled Armijo merit test in :meth:`project`.
        """

        import torch

        config = self.config
        tet_count = len(self.elements_np)
        tet_ids = torch.arange(
            tet_count, dtype=torch.long, device=self.device
        )
        ratios, volume_gradients = self.signed_volume_ratios_and_gradients(
            current
        )
        if float(torch.max(constraints)) > (
            config.sqp_strain_maximum + config.sqp_primal_tolerance
        ):
            raise RuntimeError("SQP current state violates the strain region")
        if float(torch.min(ratios)) < (
            config.sqp_minimum_signed_volume_ratio
            - config.sqp_primal_tolerance
        ):
            raise RuntimeError("SQP current state violates the volume region")

        inverse_mass_tets = self.inverse_mass[self.elements]
        strain_metric_squared = torch.sum(
            inverse_mass_tets * torch.sum(gradients * gradients, dim=2),
            dim=1,
        )
        volume_metric_squared = torch.sum(
            inverse_mass_tets
            * torch.sum(volume_gradients * volume_gradients, dim=2),
            dim=1,
        )
        valid_strain = strain_metric_squared > 1.0e-30
        if bool(torch.any(volume_metric_squared <= 1.0e-30)):
            raise RuntimeError("SQP determinant row has a zero metric norm")
        valid_strain_tets = torch.nonzero(
            valid_strain, as_tuple=False
        ).flatten()
        base_rows = torch.cat(
            (
                gradients[valid_strain_tets],
                -volume_gradients,
            ),
            dim=0,
        )
        base_tets = torch.cat((valid_strain_tets, tet_ids), dim=0)
        base_bounds = torch.cat(
            (
                config.sqp_strain_fraction_to_boundary
                * (
                    config.sqp_strain_maximum
                    - constraints[valid_strain_tets]
                ),
                config.sqp_volume_fraction_to_boundary
                * (
                    ratios - config.sqp_minimum_signed_volume_ratio
                ),
            )
        )
        # 0=current strain, 1=current determinant, 2=nonlinear strain cut,
        # 3=nonlinear determinant cut.
        base_kinds = torch.cat(
            (
                torch.zeros_like(valid_strain_tets),
                torch.ones_like(tet_ids),
            )
        )
        cut_rows: list[object] = []
        cut_tets: list[int] = []
        cut_bounds: list[float] = []
        cut_kinds: list[int] = []
        cut_fingerprints: set[str] = set()
        cut_receipts: list[dict[str, object]] = []
        resolve_receipts: list[dict[str, object]] = []
        coupling_cache: dict[int, object] = {}
        inverse_coupling_cache: dict[int, object] = {}
        auxiliary_receipt_cache: dict[int, dict[str, object]] = {}
        auxiliary_columns_computed = 0
        auxiliary_initial_linear_solves = 0
        auxiliary_refinement_linear_solves = 0
        auxiliary_linear_solves = 0
        auxiliary_pcg_iterations_total = 0
        auxiliary_zero_rhs_columns = 0

        def assemble_rows():
            if cut_rows:
                raw_rows = torch.cat(
                    (base_rows, torch.stack(cut_rows, dim=0)), dim=0
                )
                row_tets = torch.cat(
                    (
                        base_tets,
                        torch.as_tensor(
                            cut_tets,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    )
                )
                raw_bounds = torch.cat(
                    (
                        base_bounds,
                        torch.as_tensor(
                            cut_bounds,
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    )
                )
                row_kinds = torch.cat(
                    (
                        base_kinds,
                        torch.as_tensor(
                            cut_kinds,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    )
                )
            else:
                raw_rows = base_rows
                row_tets = base_tets
                raw_bounds = base_bounds
                row_kinds = base_kinds
            row_inverse_mass = self.inverse_mass[self.elements[row_tets]]
            metric_squared = torch.sum(
                row_inverse_mass * torch.sum(raw_rows * raw_rows, dim=2),
                dim=1,
            )
            if bool(torch.any(metric_squared <= 1.0e-30)):
                raise RuntimeError("SQP appended row has a zero metric norm")
            row_scale = torch.rsqrt(metric_squared)
            return (
                raw_rows * row_scale[:, None, None],
                raw_bounds * row_scale,
                row_tets,
                row_kinds,
                row_scale,
            )

        def evaluate_rays(
            kind: int,
            selected_tets: object,
            direction: object,
            fractions: object,
        ):
            selected_elements = self.elements[selected_tets]
            local_tets = (
                current[selected_elements]
                + fractions[:, None, None] * direction[selected_elements]
            )
            edge_1 = local_tets[:, 1] - local_tets[:, 0]
            edge_2 = local_tets[:, 2] - local_tets[:, 0]
            edge_3 = local_tets[:, 3] - local_tets[:, 0]
            if kind == 2:
                ds = torch.stack((edge_1, edge_2, edge_3), dim=-1)
                selected_inverse = self.rest_inverse[selected_tets]
                deformation = ds @ selected_inverse
                left, singular, right_t = torch.linalg.svd(deformation)
                reflected = torch.linalg.det(left @ right_t) < 0.0
                left = left.clone()
                singular = singular.clone()
                left[reflected, :, -1] *= -1.0
                singular[reflected, -1] *= -1.0
                delta_sigma = singular - 1.0
                values = torch.linalg.norm(delta_sigma, dim=1)
                normalized = delta_sigma / values.clamp_min(
                    self.config.constraint_epsilon
                )[:, None]
                grad_f = (left * normalized[:, None, :]) @ right_t
                grad_ds = grad_f @ selected_inverse.transpose(1, 2)
                local_gradients = torch.empty(
                    (len(selected_tets), 4, 3),
                    dtype=self.dtype,
                    device=self.device,
                )
                local_gradients[:, 1] = grad_ds[:, :, 0]
                local_gradients[:, 2] = grad_ds[:, :, 1]
                local_gradients[:, 3] = grad_ds[:, :, 2]
                local_gradients[:, 0] = -torch.sum(
                    local_gradients[:, 1:], dim=1
                )
                return values, local_gradients
            denominator = self.rest_signed_six[selected_tets, None]
            local_gradients = torch.empty(
                (len(selected_tets), 4, 3),
                dtype=self.dtype,
                device=self.device,
            )
            local_gradients[:, 1] = (
                torch.cross(edge_2, edge_3, dim=1) / denominator
            )
            local_gradients[:, 2] = (
                torch.cross(edge_3, edge_1, dim=1) / denominator
            )
            local_gradients[:, 3] = (
                torch.cross(edge_1, edge_2, dim=1) / denominator
            )
            local_gradients[:, 0] = -torch.sum(
                local_gradients[:, 1:], dim=1
            )
            values = torch.sum(
                torch.cross(edge_1, edge_2, dim=1) * edge_3, dim=1
            ) / self.rest_signed_six[selected_tets]
            return values, local_gradients

        def append_boundary_cuts(
            *,
            kind: int,
            selected_tets: object,
            rejected_direction: object,
            resolve: int,
        ) -> int:
            if len(selected_tets) == 0:
                return 0
            count = min(
                len(selected_tets), config.sqp_maximum_cuts_per_resolve
            )
            selected_tets = selected_tets[:count]
            local_direction = rejected_direction[
                self.elements[selected_tets]
            ]
            low = torch.zeros(
                count, dtype=self.dtype, device=self.device
            )
            high = torch.ones_like(low)
            found = torch.zeros(
                count, dtype=torch.bool, device=self.device
            )

            def rejected(values: object):
                if kind == 2:
                    return values > config.sqp_strain_maximum
                return values < config.sqp_minimum_signed_volume_ratio

            start_values, _start_gradients = evaluate_rays(
                kind, selected_tets, rejected_direction, low
            )
            if kind == 2:
                start_safe = start_values <= (
                    config.sqp_strain_maximum
                    + config.sqp_primal_tolerance
                )
            else:
                start_safe = start_values >= (
                    config.sqp_minimum_signed_volume_ratio
                    - config.sqp_primal_tolerance
                )
            if not bool(torch.all(start_safe)):
                raise RuntimeError("SQP boundary ray starts outside safety")
            endpoint_values, _endpoint_gradients = evaluate_rays(
                kind, selected_tets, rejected_direction, high
            )
            if not bool(torch.all(rejected(endpoint_values))):
                raise RuntimeError("SQP boundary ray endpoint is not rejected")
            for sample in range(1, config.sqp_boundary_scan_intervals + 1):
                fraction = sample / float(config.sqp_boundary_scan_intervals)
                sample_fractions = torch.full_like(low, fraction)
                sample_values, _sample_gradients = evaluate_rays(
                    kind,
                    selected_tets,
                    rejected_direction,
                    sample_fractions,
                )
                newly_found = (~found) & rejected(sample_values)
                high = torch.where(newly_found, sample_fractions, high)
                found = found | newly_found
                low = torch.where(~found, sample_fractions, low)
            if not bool(torch.all(found)):
                raise RuntimeError("SQP failed to bracket a safety boundary")
            for _iteration in range(
                config.sqp_boundary_bisection_iterations
            ):
                middle = 0.5 * (low + high)
                middle_values, _middle_gradients = evaluate_rays(
                    kind, selected_tets, rejected_direction, middle
                )
                middle_rejected = rejected(middle_values)
                high = torch.where(middle_rejected, middle, high)
                low = torch.where(middle_rejected, low, middle)
            boundary_fraction = 0.5 * (low + high)
            boundary_values, boundary_gradients = evaluate_rays(
                kind,
                selected_tets,
                rejected_direction,
                boundary_fraction,
            )
            boundary_local_direction = (
                boundary_fraction[:, None, None] * local_direction
            )
            if kind == 2:
                rows = boundary_gradients
                bounds = (
                    config.sqp_strain_maximum
                    - boundary_values
                    + torch.sum(
                        boundary_gradients * boundary_local_direction,
                        dim=(1, 2),
                    )
                )
            else:
                rows = -boundary_gradients
                bounds = (
                    boundary_values
                    - config.sqp_minimum_signed_volume_ratio
                    - torch.sum(
                        boundary_gradients * boundary_local_direction,
                        dim=(1, 2),
                    )
                )
            rejected_directional = torch.sum(
                rows * local_direction, dim=(1, 2)
            )
            if bool(
                torch.any(bounds < -config.sqp_primal_tolerance)
            ):
                raise RuntimeError(
                    "SQP boundary cut excludes the feasible zero direction"
                )
            bounds = torch.clamp_min(bounds, 0.0)
            if bool(
                torch.any(
                    rejected_directional
                    <= bounds + config.sqp_primal_tolerance
                )
            ):
                raise RuntimeError(
                    "SQP boundary cut does not exclude its rejected direction"
                )
            added = 0
            for local_index in range(count):
                row = rows[local_index].detach().clone()
                bound = float(bounds[local_index])
                tet = int(selected_tets[local_index])
                fingerprint_payload = np.concatenate(
                    (
                        np.asarray((kind, tet, bound), dtype=np.float64),
                        row.detach().cpu().numpy().astype(
                            np.float64, copy=False
                        ).reshape(-1),
                    )
                )
                fingerprint = hashlib.sha256(
                    fingerprint_payload.tobytes()
                ).hexdigest()
                if fingerprint in cut_fingerprints:
                    continue
                cut_fingerprints.add(fingerprint)
                cut_rows.append(row)
                cut_tets.append(tet)
                cut_bounds.append(bound)
                cut_kinds.append(kind)
                cut_receipts.append(
                    {
                        "resolve": int(resolve),
                        "kind": (
                            "nonlinear_strain_boundary"
                            if kind == 2
                            else "nonlinear_determinant_boundary"
                        ),
                        "tetrahedron": tet,
                        "boundary_fraction": float(
                            boundary_fraction[local_index]
                        ),
                        "boundary_value": float(
                            boundary_values[local_index]
                        ),
                        "bound": bound,
                        "exclusion_margin": float(
                            rejected_directional[local_index]
                            - bounds[local_index]
                        ),
                        "zero_direction_feasible": True,
                    }
                )
                added += 1
            return added

        def solve_linearized_qp(
            rows: object,
            bounds: object,
            row_tets: object,
            row_kinds: object,
        ) -> dict[str, object]:
            nonlocal auxiliary_columns_computed
            nonlocal auxiliary_initial_linear_solves
            nonlocal auxiliary_refinement_linear_solves
            nonlocal auxiliary_linear_solves
            nonlocal auxiliary_pcg_iterations_total
            nonlocal auxiliary_zero_rhs_columns

            def all_directional(direction: object):
                return self._local_rows_directional(
                    rows, direction, row_tets
                )

            direction = torch.zeros_like(base_direction)
            material_adjustment = torch.zeros_like(base_delta_lambda)
            multipliers64 = torch.empty(
                0, dtype=torch.float64, device=self.device
            )
            active: list[int] = []
            additions = 0
            removals = 0
            converged = False
            final_schur64 = torch.empty(
                (0, 0), dtype=torch.float64, device=self.device
            )
            final_raw_schur64 = final_schur64
            final_schur_rhs64 = torch.empty(
                0, dtype=torch.float64, device=self.device
            )
            final_coupling = torch.empty(
                (tet_count, 0), dtype=self.dtype, device=self.device
            )
            final_inverse_coupling = torch.empty_like(final_coupling)
            schur_asymmetry_relative = 0.0
            schur_minimum_eigenvalue = 0.0
            schur_condition_number = 1.0
            for active_iteration in range(
                1, config.sqp_maximum_active_set_iterations + 1
            ):
                if active:
                    active_tensor = torch.as_tensor(
                        active, dtype=torch.long, device=self.device
                    )
                    active_rows = rows[active_tensor]
                    active_tets = row_tets[active_tensor]
                    missing = [
                        index
                        for index in active
                        if index not in coupling_cache
                    ]
                    for index in missing:
                        single_row = rows[index : index + 1]
                        single_tet = row_tets[index : index + 1]
                        weighted_position = self._local_rows_transpose(
                            single_row,
                            torch.ones(
                                1, dtype=self.dtype, device=self.device
                            ),
                            single_tet,
                        )
                        weighted_position *= self.inverse_mass[:, None]
                        coupling_rhs = self._local_rows_directional(
                            gradients, weighted_position
                        )
                        rhs_l2 = float(torch.linalg.norm(coupling_rhs))
                        if rhs_l2 <= 1.0e-20:
                            inverse_coupling = torch.zeros_like(coupling_rhs)
                            aux_iterations = 0
                            aux_recursive = 0.0
                            aux_true = 0.0
                            aux_operator_error = 0.0
                            aux_replacements = 0
                            aux_refinements = 0
                            refinement_true_residual_history = [0.0]
                            zero_rhs = True
                        else:
                            (
                                inverse_coupling,
                                aux_iterations,
                                aux_recursive,
                                _reported_aux_true,
                                aux_operator_error,
                                aux_replacements,
                            ) = self._pcg(
                                coupling_rhs, gradients, diagonal
                            )
                            true_residual = coupling_rhs - self._operator(
                                inverse_coupling, gradients
                            )
                            aux_true = float(
                                torch.linalg.norm(true_residual) / rhs_l2
                            )
                            aux_refinements = 0
                            refinement_true_residual_history = [
                                float(aux_true)
                            ]
                            while (
                                aux_true
                                > config.sqp_auxiliary_relative_residual
                                and aux_refinements
                                < config.sqp_maximum_auxiliary_refinements
                            ):
                                (
                                    refinement,
                                    refinement_iterations,
                                    refinement_recursive,
                                    _refinement_reported_true,
                                    refinement_operator_error,
                                    refinement_replacements,
                                ) = self._pcg(
                                    true_residual, gradients, diagonal
                                )
                                inverse_coupling = (
                                    inverse_coupling + refinement
                                )
                                aux_iterations += refinement_iterations
                                aux_recursive = refinement_recursive
                                aux_operator_error = max(
                                    aux_operator_error,
                                    refinement_operator_error,
                                )
                                aux_replacements += refinement_replacements
                                aux_refinements += 1
                                true_residual = (
                                    coupling_rhs
                                    - self._operator(
                                        inverse_coupling, gradients
                                    )
                                )
                                aux_true = float(
                                    torch.linalg.norm(true_residual) / rhs_l2
                                )
                                refinement_true_residual_history.append(
                                    float(aux_true)
                                )
                            if (
                                aux_true
                                > config.sqp_auxiliary_relative_residual
                            ):
                                raise RuntimeError(
                                    "SQP auxiliary iterative refinement did "
                                    "not reach its true-residual target: "
                                    f"{aux_true:.9g} > "
                                    f"{config.sqp_auxiliary_relative_residual:.9g}; "
                                    "true_residual_history="
                                    f"{refinement_true_residual_history}"
                                )
                            zero_rhs = False
                        auxiliary_columns_computed += 1
                        auxiliary_pcg_iterations_total += int(aux_iterations)
                        if zero_rhs:
                            auxiliary_zero_rhs_columns += 1
                            column_linear_solves = 0
                        else:
                            auxiliary_initial_linear_solves += 1
                            auxiliary_refinement_linear_solves += int(
                                aux_refinements
                            )
                            column_linear_solves = 1 + int(aux_refinements)
                            auxiliary_linear_solves += column_linear_solves
                        coupling_cache[index] = coupling_rhs
                        inverse_coupling_cache[index] = inverse_coupling
                        auxiliary_receipt_cache[index] = {
                            "row_index": int(index),
                            "kind": int(row_kinds[index]),
                            "tetrahedron": int(row_tets[index]),
                            "rhs_l2": rhs_l2,
                            "pcg_iterations": int(aux_iterations),
                            "recursive_relative_residual": float(
                                aux_recursive
                            ),
                            "true_relative_residual": float(aux_true),
                            "level0_physical_operator_relative_error": float(
                                aux_operator_error
                            ),
                            "residual_replacements": int(aux_replacements),
                            "iterative_refinements": int(aux_refinements),
                            "pcg_solve_calls": int(column_linear_solves),
                            "refinement_true_residual_history": (
                                refinement_true_residual_history
                            ),
                            "zero_rhs_skipped": bool(zero_rhs),
                        }
                    coupling = torch.stack(
                        [coupling_cache[index] for index in active], dim=1
                    )
                    inverse_coupling = torch.stack(
                        [
                            inverse_coupling_cache[index]
                            for index in active
                        ],
                        dim=1,
                    )
                    direct_gram64 = self._local_rows_mass_gram(
                        active_rows, active_tets
                    )
                    raw_schur64 = direct_gram64 - (
                        coupling.to(torch.float64).transpose(0, 1)
                        @ inverse_coupling.to(torch.float64)
                    )
                    schur64 = 0.5 * (
                        raw_schur64 + raw_schur64.transpose(0, 1)
                    )
                    schur_rhs64 = (
                        all_directional(base_direction)[active_tensor]
                        - bounds[active_tensor]
                    ).to(torch.float64)
                    cholesky64, cholesky_info = torch.linalg.cholesky_ex(
                        schur64
                    )
                    if bool(torch.any(cholesky_info != 0)):
                        raise RuntimeError(
                            "SQP active Schur Cholesky failed"
                        )
                    multipliers64 = torch.cholesky_solve(
                        schur_rhs64[:, None], cholesky64
                    )[:, 0]
                    target_material_adjustment = (
                        inverse_coupling
                        @ multipliers64.to(self.dtype)
                    )
                    safety_impulse = self._local_rows_transpose(
                        active_rows,
                        multipliers64.to(self.dtype),
                        active_tets,
                    )
                    material_impulse = self._local_rows_transpose(
                        gradients, target_material_adjustment
                    )
                    target = base_direction + self.inverse_mass[:, None] * (
                        material_impulse - safety_impulse
                    )
                    final_schur64 = schur64
                    final_raw_schur64 = raw_schur64
                    final_schur_rhs64 = schur_rhs64
                    final_coupling = coupling
                    final_inverse_coupling = inverse_coupling
                else:
                    target = base_direction
                    target_material_adjustment = torch.zeros_like(
                        base_delta_lambda
                    )
                    multipliers64 = torch.empty(
                        0, dtype=torch.float64, device=self.device
                    )
                search = target - direction
                directional = all_directional(search)
                slack = bounds - all_directional(direction)
                inactive = torch.ones(
                    len(bounds), dtype=torch.bool, device=self.device
                )
                if active:
                    inactive[
                        torch.as_tensor(
                            active, dtype=torch.long, device=self.device
                        )
                    ] = False
                blocking = inactive & (
                    directional
                    > slack + config.sqp_primal_tolerance
                )
                if bool(torch.any(blocking)):
                    fractions = torch.where(
                        blocking,
                        torch.clamp(
                            slack / directional.clamp_min(1.0e-30),
                            0.0,
                            1.0,
                        ),
                        torch.full_like(slack, float("inf")),
                    )
                    active_step, blocking_index = torch.min(
                        fractions, dim=0
                    )
                    direction = direction + active_step * search
                    material_adjustment = material_adjustment + active_step * (
                        target_material_adjustment - material_adjustment
                    )
                    if len(active) >= config.sqp_maximum_active_constraints:
                        raise RuntimeError("SQP active set reached its limit")
                    active.append(int(blocking_index))
                    additions += 1
                    continue
                direction = target
                material_adjustment = target_material_adjustment
                negative = torch.nonzero(
                    multipliers64 < -config.sqp_dual_tolerance,
                    as_tuple=False,
                ).flatten()
                if len(negative):
                    local_remove = int(
                        negative[
                            torch.argmin(multipliers64[negative])
                        ]
                    )
                    del active[local_remove]
                    removals += 1
                    continue
                final_violation = all_directional(direction) - bounds
                if float(torch.max(final_violation)) <= (
                    config.sqp_primal_tolerance
                ):
                    converged = True
                    break
            else:
                active_iteration = config.sqp_maximum_active_set_iterations
            if not converged:
                raise RuntimeError("SQP active set did not converge")

            # Cholesky is the per-iteration SPD test needed by the solver.
            # The spectral/asymmetry values are diagnostics only, so compute
            # them once for the final working set instead of synchronizing the
            # Radeon device on every active-set add/remove iteration.
            if active:
                schur_asymmetry_relative = float(
                    torch.linalg.norm(
                        final_raw_schur64
                        - final_raw_schur64.transpose(0, 1)
                    )
                    / torch.linalg.norm(final_raw_schur64).clamp_min(1.0e-30)
                )
                final_eigenvalues64 = torch.linalg.eigvalsh(final_schur64)
                schur_minimum_eigenvalue = float(
                    torch.min(final_eigenvalues64)
                )
                maximum_eigenvalue = float(
                    torch.max(final_eigenvalues64)
                )
                if (
                    not np.isfinite(schur_minimum_eigenvalue)
                    or schur_minimum_eigenvalue <= 0.0
                ):
                    raise RuntimeError(
                        "SQP final active Schur complement is not positive "
                        "definite"
                    )
                schur_condition_number = (
                    maximum_eigenvalue / schur_minimum_eigenvalue
                )

            adjusted_delta_lambda = (
                base_delta_lambda + material_adjustment
            )
            active_tensor = torch.as_tensor(
                active, dtype=torch.long, device=self.device
            )
            final_violation = all_directional(direction) - bounds
            primal_violation = max(float(torch.max(final_violation)), 0.0)
            active_residual = (
                final_violation[active_tensor]
                if active
                else torch.empty(0, dtype=self.dtype, device=self.device)
            )
            active_equality = (
                float(torch.max(torch.abs(active_residual)))
                if active
                else 0.0
            )
            minimum_multiplier = (
                float(torch.min(multipliers64)) if active else 0.0
            )
            dual_violation = max(-minimum_multiplier, 0.0)
            complementarity = (
                float(
                    torch.max(
                        torch.abs(
                            multipliers64.to(self.dtype) * active_residual
                        )
                    )
                )
                if active
                else 0.0
            )
            displacement = direction - base_direction
            mass_term = displacement / self.inverse_mass[:, None]
            material_displacement = self._local_rows_directional(
                gradients, displacement
            )
            material_term = self._local_rows_transpose(
                gradients,
                material_displacement / self.alpha_tilde,
            )
            safety_term = (
                self._local_rows_transpose(
                    rows[active_tensor],
                    multipliers64.to(self.dtype),
                    row_tets[active_tensor],
                )
                if active
                else torch.zeros_like(direction)
            )
            stationarity = mass_term + material_term + safety_term
            stationarity_denominator = (
                torch.linalg.norm(mass_term)
                + torch.linalg.norm(material_term)
                + torch.linalg.norm(safety_term)
            ).clamp_min(1.0e-30)
            stationarity_relative = float(
                torch.linalg.norm(stationarity)
                / stationarity_denominator
            )
            # Reconstruct the same KKT stationarity balance in FP64 from the
            # Schur variables instead of subtracting ``direction -
            # base_direction`` in FP32.  The latter loses the small safety
            # correction when the unconstrained MGPBD direction is much
            # larger.  Keep both values in the receipt so the accepted gate
            # remains independently auditable.
            stationarity_reconstructed_fp64_relative = 0.0
            if active:
                gradients64 = gradients.to(torch.float64)
                active_rows64 = rows[active_tensor].to(torch.float64)
                material_adjustment64 = material_adjustment.to(torch.float64)
                inverse_mass64 = self.inverse_mass.to(torch.float64)
                alpha64 = self.alpha_tilde.to(torch.float64)
                material_impulse64 = torch.zeros(
                    self.rest_positions.shape,
                    dtype=torch.float64,
                    device=self.device,
                )
                safety_impulse64 = torch.zeros_like(material_impulse64)
                weighted_material64 = (
                    material_adjustment64[:, None, None] * gradients64
                )
                weighted_safety64 = (
                    multipliers64[:, None, None] * active_rows64
                )
                active_elements = self.elements[row_tets[active_tensor]]
                for local_index in range(4):
                    material_impulse64.index_add_(
                        0,
                        self.elements[:, local_index],
                        weighted_material64[:, local_index],
                    )
                    safety_impulse64.index_add_(
                        0,
                        active_elements[:, local_index],
                        weighted_safety64[:, local_index],
                    )
                reconstructed_mass_term64 = (
                    material_impulse64 - safety_impulse64
                )
                reconstructed_displacement64 = (
                    inverse_mass64[:, None] * reconstructed_mass_term64
                )
                reconstructed_material_displacement64 = torch.sum(
                    gradients64
                    * reconstructed_displacement64[self.elements],
                    dim=(1, 2),
                )
                reconstructed_material_term64 = torch.zeros_like(
                    material_impulse64
                )
                weighted_reconstructed_material64 = (
                    reconstructed_material_displacement64 / alpha64
                )[:, None, None] * gradients64
                for local_index in range(4):
                    reconstructed_material_term64.index_add_(
                        0,
                        self.elements[:, local_index],
                        weighted_reconstructed_material64[:, local_index],
                    )
                reconstructed_stationarity64 = (
                    reconstructed_mass_term64
                    + reconstructed_material_term64
                    + safety_impulse64
                )
                reconstructed_denominator64 = (
                    torch.linalg.norm(reconstructed_mass_term64)
                    + torch.linalg.norm(reconstructed_material_term64)
                    + torch.linalg.norm(safety_impulse64)
                ).clamp_min(1.0e-30)
                stationarity_reconstructed_fp64_relative = float(
                    torch.linalg.norm(reconstructed_stationarity64)
                    / reconstructed_denominator64
                )
            schur_residual_relative = 0.0
            auxiliary_combination_relative = 0.0
            schur_solve_residual_l2 = 0.0
            if active:
                schur_residual64 = (
                    final_schur64 @ multipliers64
                    - final_schur_rhs64
                )
                schur_solve_residual_l2 = float(
                    torch.linalg.norm(schur_residual64)
                )
                schur_residual_relative = float(
                    torch.linalg.norm(schur_residual64)
                    / (
                        torch.linalg.norm(final_schur_rhs64)
                        + torch.linalg.norm(final_schur64)
                        * torch.linalg.norm(multipliers64)
                    ).clamp_min(1.0e-30)
                )
                weighted_coupling = (
                    final_coupling
                    @ multipliers64.to(self.dtype)
                )
                auxiliary_action = self._operator(
                    material_adjustment, gradients
                )
                auxiliary_combination_relative = float(
                    torch.linalg.norm(weighted_coupling - auxiliary_action)
                    / torch.linalg.norm(weighted_coupling).clamp_min(1.0e-30)
                )
            linearized_direction = (
                self._local_rows_directional(gradients, direction)
                + self.alpha_tilde * adjusted_delta_lambda
            )
            coupled_residual = material_residual + linearized_direction
            coupled_relative = float(
                torch.linalg.norm(coupled_residual)
                / torch.linalg.norm(material_residual).clamp_min(1.0e-30)
            )
            merit_slope = float(
                torch.dot(material_residual, linearized_direction)
            )
            auxiliary_receipts = [
                auxiliary_receipt_cache[index] for index in active
            ]
            maximum_auxiliary = max(
                (
                    float(receipt["true_relative_residual"])
                    for receipt in auxiliary_receipts
                ),
                default=0.0,
            )
            kkt_checks = {
                "primal_feasible": (
                    primal_violation <= config.sqp_primal_tolerance
                ),
                "dual_feasible": (
                    dual_violation <= config.sqp_dual_tolerance
                ),
                "active_equalities_satisfied": (
                    active_equality <= config.sqp_kkt_relative_tolerance
                ),
                "complementarity_satisfied": (
                    complementarity
                    <= config.sqp_kkt_relative_tolerance
                ),
                "stationarity_satisfied": (
                    stationarity_relative
                    <= config.sqp_kkt_relative_tolerance
                ),
                "schur_system_satisfied": (
                    schur_residual_relative
                    <= config.sqp_kkt_relative_tolerance
                ),
                "auxiliary_solves_satisfied": (
                    maximum_auxiliary
                    <= config.sqp_auxiliary_relative_residual * 1.05
                ),
                "coupled_material_residual_satisfied": (
                    coupled_relative
                    <= config.sqp_coupled_relative_tolerance
                ),
                "auxiliary_combination_satisfied": (
                    auxiliary_combination_relative
                    <= config.sqp_coupled_relative_tolerance
                ),
                "merit_descent": (
                    np.isfinite(merit_slope) and merit_slope < 0.0
                ),
                "schur_positive_definite": (
                    not active or schur_minimum_eigenvalue > 0.0
                ),
            }
            if not all(kkt_checks.values()):
                diagnostic_values = {
                    "primal_feasible": primal_violation,
                    "dual_feasible": dual_violation,
                    "active_equalities_satisfied": active_equality,
                    "complementarity_satisfied": complementarity,
                    "stationarity_satisfied": stationarity_relative,
                    "stationarity_reconstructed_fp64": (
                        stationarity_reconstructed_fp64_relative
                    ),
                    "schur_system_satisfied": schur_residual_relative,
                    "auxiliary_solves_satisfied": maximum_auxiliary,
                    "coupled_material_residual_satisfied": coupled_relative,
                    "auxiliary_combination_satisfied": (
                        auxiliary_combination_relative
                    ),
                    "merit_descent": merit_slope,
                    "schur_positive_definite": schur_minimum_eigenvalue,
                }
                failed = sorted(
                    f"{name}={diagnostic_values[name]:.9g}"
                    for name, passed in kkt_checks.items()
                    if not passed
                )
                raise RuntimeError(
                    "SQP direction failed its KKT gate: "
                    + ", ".join(failed)
                    + "; stationarity_reconstructed_fp64="
                    + f"{stationarity_reconstructed_fp64_relative:.9g}"
                )
            return {
                "direction": direction,
                "adjusted_delta_lambda": adjusted_delta_lambda,
                "active": active,
                "active_tets": row_tets[active_tensor],
                "active_kinds": row_kinds[active_tensor],
                "multipliers": multipliers64,
                "iterations": int(active_iteration),
                "additions": int(additions),
                "removals": int(removals),
                "final_maximum_linearized_violation": float(
                    torch.max(final_violation)
                ),
                "minimum_multiplier": minimum_multiplier,
                "active_equality_residual_maximum": active_equality,
                "complementarity_maximum": complementarity,
                "stationarity_relative": stationarity_relative,
                "stationarity_reconstructed_fp64_relative": (
                    stationarity_reconstructed_fp64_relative
                ),
                "schur_residual_relative": schur_residual_relative,
                "schur_solve_residual_l2": schur_solve_residual_l2,
                "schur_minimum_eigenvalue": schur_minimum_eigenvalue,
                "schur_condition_number": schur_condition_number,
                "schur_asymmetry_relative": schur_asymmetry_relative,
                "maximum_auxiliary_true_relative_residual": (
                    maximum_auxiliary
                ),
                "auxiliary_combination_residual_relative": (
                    auxiliary_combination_relative
                ),
                "auxiliary_receipts": auxiliary_receipts,
                "coupled_linearized_residual_relative": coupled_relative,
                "coupled_linearized_residual_l2": float(
                    torch.linalg.norm(coupled_residual)
                ),
                "merit_slope": merit_slope,
                "kkt_checks": kkt_checks,
                "kkt_passed": True,
            }

        raw_positions = current + base_direction
        raw_constraints, _raw_gradients, _raw_active = (
            self.constraints_and_gradients(raw_positions)
        )
        raw_ratios = self.signed_volume_ratios(raw_positions)
        progress_enabled = os.environ.get(
            "ONELOOP_MGPBD_SQP_PROGRESS", "0"
        ) == "1"

        def emit_resolve_progress(
            receipt: dict[str, object], qp: dict[str, object]
        ) -> None:
            if not progress_enabled:
                return
            print(
                "MGPBD_SQP_PROGRESS "
                + json.dumps(
                    {
                        **receipt,
                        "active_set_iterations": int(qp["iterations"]),
                        "stationarity_relative": float(
                            qp["stationarity_relative"]
                        ),
                        "maximum_auxiliary_true_relative_residual": float(
                            qp[
                                "maximum_auxiliary_true_relative_residual"
                            ]
                        ),
                        "auxiliary_pcg_calls_total": int(
                            auxiliary_linear_solves
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        final_qp: dict[str, object] | None = None
        final_constraints: object | None = None
        final_ratios: object | None = None
        for resolve in range(
            1, config.sqp_maximum_nonlinear_cut_resolves + 1
        ):
            rows, bounds, row_tets, row_kinds, _row_scale = assemble_rows()
            qp = solve_linearized_qp(rows, bounds, row_tets, row_kinds)
            candidate_positions = current + qp["direction"]
            candidate_constraints, _candidate_gradients, _candidate_active = (
                self.constraints_and_gradients(candidate_positions)
            )
            candidate_ratios = self.signed_volume_ratios(candidate_positions)
            strain_violation = (
                candidate_constraints - config.sqp_strain_maximum
            )
            determinant_violation = (
                config.sqp_minimum_signed_volume_ratio - candidate_ratios
            )
            strain_bad = torch.nonzero(
                strain_violation > 0.0, as_tuple=False
            ).flatten()
            determinant_bad = torch.nonzero(
                determinant_violation > 0.0, as_tuple=False
            ).flatten()
            receipt: dict[str, object] = {
                "resolve": int(resolve),
                "linearized_rows": int(len(bounds)),
                "active_constraints": int(len(qp["active"])),
                "strain_violation_count": int(len(strain_bad)),
                "strain_violation_maximum": max(
                    float(torch.max(strain_violation)), 0.0
                ),
                "determinant_violation_count": int(len(determinant_bad)),
                "determinant_violation_maximum": max(
                    float(torch.max(determinant_violation)), 0.0
                ),
            }
            if not len(strain_bad) and not len(determinant_bad):
                receipt["full_direction_safety_feasible"] = True
                receipt["new_strain_cuts"] = 0
                receipt["new_determinant_cuts"] = 0
                resolve_receipts.append(receipt)
                emit_resolve_progress(receipt, qp)
                final_qp = qp
                final_constraints = candidate_constraints
                final_ratios = candidate_ratios
                break
            remaining = config.sqp_maximum_cuts_per_resolve
            new_strain = 0
            new_determinant = 0
            if len(strain_bad):
                ordered_strain = strain_bad[
                    torch.argsort(
                        strain_violation[strain_bad], descending=True
                    )
                ]
                reserve_determinant = 1 if len(determinant_bad) else 0
                strain_limit = max(remaining - reserve_determinant, 0)
                if strain_limit:
                    new_strain = append_boundary_cuts(
                        kind=2,
                        selected_tets=ordered_strain[:strain_limit],
                        rejected_direction=qp["direction"],
                        resolve=resolve,
                    )
                    remaining -= new_strain
            if len(determinant_bad) and remaining > 0:
                ordered_determinant = determinant_bad[
                    torch.argsort(
                        determinant_violation[determinant_bad],
                        descending=True,
                    )
                ]
                new_determinant = append_boundary_cuts(
                    kind=3,
                    selected_tets=ordered_determinant[:remaining],
                    rejected_direction=qp["direction"],
                    resolve=resolve,
                )
            receipt["full_direction_safety_feasible"] = False
            receipt["new_strain_cuts"] = int(new_strain)
            receipt["new_determinant_cuts"] = int(new_determinant)
            resolve_receipts.append(receipt)
            emit_resolve_progress(receipt, qp)
            if new_strain + new_determinant == 0:
                raise RuntimeError(
                    "SQP nonlinear safety separation produced no new cut"
                )
        if final_qp is None or final_constraints is None or final_ratios is None:
            raise RuntimeError("SQP nonlinear cutting-plane limit exhausted")

        direction = final_qp["direction"]
        adjusted_delta_lambda = final_qp["adjusted_delta_lambda"]
        active_kinds = final_qp["active_kinds"]
        active_tets = final_qp["active_tets"]
        active_pairs = np.column_stack(
            (
                active_kinds.detach().cpu().numpy().astype(np.int64),
                active_tets.detach().cpu().numpy().astype(np.int64),
            )
        )
        active_digest = hashlib.sha256(active_pairs.tobytes()).hexdigest()
        base_norm = torch.linalg.norm(base_direction).clamp_min(1.0e-30)
        constrained_norm = torch.linalg.norm(direction).clamp_min(1.0e-30)
        cosine = float(
            torch.sum(base_direction * direction)
            / (base_norm * constrained_norm)
        )
        metrics: dict[str, object] = {
            "enabled": True,
            "backend": "Torch_ROCm_MGPCG_Schur_active_set",
            "converged": True,
            "fallback_used": False,
            "configuration": {
                "strain_maximum": config.sqp_strain_maximum,
                "minimum_signed_volume_ratio": (
                    config.sqp_minimum_signed_volume_ratio
                ),
                "strain_fraction_to_boundary": (
                    config.sqp_strain_fraction_to_boundary
                ),
                "volume_fraction_to_boundary": (
                    config.sqp_volume_fraction_to_boundary
                ),
                "primal_tolerance": config.sqp_primal_tolerance,
                "dual_tolerance": config.sqp_dual_tolerance,
                "kkt_relative_tolerance": (
                    config.sqp_kkt_relative_tolerance
                ),
                "coupled_relative_tolerance": (
                    config.sqp_coupled_relative_tolerance
                ),
                "auxiliary_relative_residual_tolerance": (
                    config.sqp_auxiliary_relative_residual
                ),
                "maximum_auxiliary_refinements": (
                    config.sqp_maximum_auxiliary_refinements
                ),
            },
            "active_set_iterations": int(final_qp["iterations"]),
            "active_constraints": int(len(final_qp["active"])),
            "active_current_strain_constraints": int(
                torch.count_nonzero(active_kinds == 0)
            ),
            "active_determinant_constraints": int(
                torch.count_nonzero((active_kinds == 1) | (active_kinds == 3))
            ),
            "active_nonlinear_strain_cuts": int(
                torch.count_nonzero(active_kinds == 2)
            ),
            "active_set_additions": int(final_qp["additions"]),
            "active_set_removals": int(final_qp["removals"]),
            "active_pairs_head": active_pairs[:32].tolist(),
            "active_pairs_sha256": active_digest,
            "inactive_zero_gradient_strain_constraints": int(
                torch.count_nonzero(~valid_strain)
            ),
            "monitored_strain_constraints": int(
                torch.count_nonzero(valid_strain)
            ),
            "monitored_determinant_constraints": tet_count,
            "determinant_constraints_inside_activation_band": int(
                torch.count_nonzero(
                    ratios <= config.sqp_determinant_activation_ratio
                )
            ),
            "excluded_determinant_constraints": 0,
            "nonlinear_safety_resolves": int(len(resolve_receipts)),
            "nonlinear_strain_cuts": int(
                sum(kind == 2 for kind in cut_kinds)
            ),
            "nonlinear_determinant_cuts": int(
                sum(kind == 3 for kind in cut_kinds)
            ),
            "full_direction_safety_feasible": True,
            "nonlinear_resolve_receipts": resolve_receipts,
            "boundary_cut_receipts": cut_receipts,
            "full_direction_arap_maximum": float(
                torch.max(final_constraints)
            ),
            "full_direction_minimum_signed_volume_ratio": float(
                torch.min(final_ratios)
            ),
            "raw_full_candidate_arap_maximum": float(
                torch.max(raw_constraints)
            ),
            "raw_full_candidate_minimum_signed_volume_ratio": float(
                torch.min(raw_ratios)
            ),
            "base_to_constrained_direction_cosine": cosine,
            "direction_change_l2": float(
                torch.linalg.norm(direction - base_direction)
            ),
            "final_maximum_linearized_violation": float(
                final_qp["final_maximum_linearized_violation"]
            ),
            "minimum_multiplier": float(final_qp["minimum_multiplier"]),
            "active_equality_residual_maximum": float(
                final_qp["active_equality_residual_maximum"]
            ),
            "complementarity_maximum": float(
                final_qp["complementarity_maximum"]
            ),
            "stationarity_relative": float(
                final_qp["stationarity_relative"]
            ),
            "schur_residual_relative": float(
                final_qp["schur_residual_relative"]
            ),
            "schur_solve_residual_l2": float(
                final_qp["schur_solve_residual_l2"]
            ),
            "schur_minimum_eigenvalue": float(
                final_qp["schur_minimum_eigenvalue"]
            ),
            "schur_condition_number": float(
                final_qp["schur_condition_number"]
            ),
            "schur_asymmetry_relative": float(
                final_qp["schur_asymmetry_relative"]
            ),
            "maximum_auxiliary_true_relative_residual": float(
                final_qp["maximum_auxiliary_true_relative_residual"]
            ),
            "auxiliary_combination_residual_relative": float(
                final_qp["auxiliary_combination_residual_relative"]
            ),
            "auxiliary_linear_solves": int(auxiliary_linear_solves),
            "auxiliary_columns_computed": int(auxiliary_columns_computed),
            "auxiliary_initial_linear_solves": int(
                auxiliary_initial_linear_solves
            ),
            "auxiliary_refinement_linear_solves": int(
                auxiliary_refinement_linear_solves
            ),
            "auxiliary_pcg_iterations_total": int(
                auxiliary_pcg_iterations_total
            ),
            "auxiliary_zero_rhs_columns": int(auxiliary_zero_rhs_columns),
            "auxiliary_final_active_columns": int(
                len(final_qp["auxiliary_receipts"])
            ),
            "auxiliary_receipts": final_qp["auxiliary_receipts"],
            "coupled_linearized_residual_relative": float(
                final_qp["coupled_linearized_residual_relative"]
            ),
            "coupled_linearized_residual_l2": float(
                final_qp["coupled_linearized_residual_l2"]
            ),
            "linearized_material_residual_l2": float(
                final_qp["coupled_linearized_residual_l2"]
            ),
            "merit_slope": float(final_qp["merit_slope"]),
            "kkt_checks": final_qp["kkt_checks"],
            "kkt_passed": bool(final_qp["kkt_passed"]),
        }
        return direction, adjusted_delta_lambda, metrics

    def project(
        self,
        positions: object,
        *,
        post_iteration: Callable[[object], object] | None = None,
    ):
        """Project predicted positions and optionally apply collision each outer step."""

        import torch

        if post_iteration is not None and (
            self.config.soc_admm_direction_enabled
            or self.config.sqp_direction_enabled
        ):
            raise ValueError(
                "constrained MGPBD cannot apply a post-iteration callback "
                "after its audited atomic line search; contact must be part "
                "of the trial state and re-audited before commit"
            )

        started = time.perf_counter()
        current = positions.detach().reshape(-1, 3).clone()
        self._last_accepted_outer_positions = current.detach().clone()
        self._last_accepted_outer_iteration = 0
        lagrangian = torch.zeros(
            len(self.elements_np), dtype=self.dtype, device=self.device
        )
        total_pcg = 0
        total_soc_admm_iterations = 0
        total_soc_admm_pcg = 0
        final_recursive_relative = 0.0
        final_true_relative = 0.0
        final_level0_operator_error = 0.0
        initial_max = 0.0
        hierarchy_build_ms = 0.0
        diagonal_dynamic_range = 1.0
        line_search_reductions = 0
        outer_history: list[dict[str, object]] = []
        initial_dual_norm: float | None = None
        project_rap_refreshes_before = self._rap_numeric_refreshes
        for outer in range(self.config.nonlinear_iterations):
            constraints, gradients, active = self.constraints_and_gradients(current)
            orientation_record: dict[str, float | int | bool] = {}
            if self.config.orientation_diagnostics_enabled:
                current_volume_ratios = self.signed_volume_ratios(current)
                orientation_record.update(
                    {
                        "current_minimum_signed_volume_ratio": float(
                            torch.min(current_volume_ratios)
                        ),
                        "current_inverted_tetrahedra": int(
                            torch.count_nonzero(current_volume_ratios < 0.0)
                        ),
                        "current_collapsed_tetrahedra": int(
                            torch.count_nonzero(current_volume_ratios <= 1.0e-8)
                        ),
                    }
                )
            if outer == 0:
                initial_max = float(torch.max(constraints))
            active_count = int(torch.count_nonzero(active))
            outer_rap_refreshes_before = self._rap_numeric_refreshes
            material_residual = constraints + self.alpha_tilde * lagrangian
            direction_backend = "mgpbd_pcg"
            legacy_direction_pcg_skipped = False
            amg_ready_for_solve = False
            rap_refreshed_for_current_matrix = False
            fine_diagonal_sum: float | None = None
            fine_diagonal_l2: float | None = None
            iterations = 0
            residual_replacements = 0
            sqp_direction_metrics: dict[str, object] | None = None
            soc_admm_direction_metrics: dict[str, object] | None = None
            if self.config.soc_admm_direction_enabled:
                # SOC-ADMM is a complete constrained Gauss--Newton direction,
                # not a filter on the legacy PCG result.  In particular this
                # branch must not build/refresh dual AMG, solve the old system,
                # or enter the retired explicit active-set path.
                direction_backend = "soc_admm"
                legacy_direction_pcg_skipped = True
                from sim.genesis_so101.mgpbd_soc_admm import (
                    SOCADMMConvergenceError,
                    solve_soc_admm_direction,
                )

                try:
                    result = solve_soc_admm_direction(
                        current=current,
                        rest_inverse=self.rest_inverse,
                        elements=self.elements,
                        masses=torch.reciprocal(self.inverse_mass),
                        material_gradients=gradients,
                        q=material_residual,
                        alpha=self.alpha_tilde,
                        config=self.config.soc_admm_config(),
                    )
                except SOCADMMConvergenceError as error:
                    # No projector state has been committed at this point.
                    # Preserve the solver's complete partial receipt both on
                    # the exception and on the projector for postmortem gates.
                    self.last_metrics = {
                        "available": True,
                        "projection_failed": True,
                        "failure": "soc_admm_direction_failed",
                        "failed_outer_iteration": outer + 1,
                        "projected_frames": self._frame,
                        "constraint_kind": (
                            "one_ARAP_singular_value_norm_per_tetrahedron"
                        ),
                        "direction_backend": "soc_admm",
                        "legacy_direction_pcg_skipped": True,
                        "arap_maximum_at_failure": float(
                            torch.max(constraints)
                        ),
                        "soc_admm_direction": error.receipt,
                        "completed_outer_iterations": outer_history,
                        "last_accepted_outer_iteration": int(
                            self._last_accepted_outer_iteration
                        ),
                        "project_ms_at_failure": (
                            time.perf_counter() - started
                        )
                        * 1000.0,
                        "configuration": self.config.to_dict(),
                    }
                    raise
                correction = result.direction
                delta_lambda = result.delta_lambda
                linearized_material = (
                    self._local_rows_directional(gradients, correction)
                    + self.alpha_tilde * delta_lambda
                )
                merit_slope = float(
                    torch.dot(material_residual, linearized_material)
                )
                soc_admm_direction_metrics = {
                    **result.metrics,
                    "enabled": True,
                    "merit_slope": merit_slope,
                }
                total_soc_admm_iterations += int(
                    result.metrics["admm_iterations"]
                )
                total_soc_admm_pcg += int(
                    result.metrics["pcg_iterations_total"]
                )
            else:
                rhs = -material_residual
                diagonal = torch.sum(
                    self.inverse_mass[self.elements]
                    * torch.sum(gradients * gradients, dim=2),
                    dim=1,
                ) + self.alpha_tilde
                fine_diagonal_sum = float(torch.sum(diagonal))
                fine_diagonal_l2 = float(torch.linalg.norm(diagonal))
                diagonal_dynamic_range = max(
                    diagonal_dynamic_range,
                    float(torch.max(diagonal) / torch.min(diagonal)),
                )
                # The ARAP gradient is intentionally undefined at exact rest.
                # Build the dual hierarchy only for the legacy direction.
                minimum_active = max(
                    4,
                    int(
                        np.ceil(
                            self.config.amg_min_active_fraction
                            * len(self.elements_np)
                        )
                    ),
                )
                if (
                    active_count >= minimum_active
                    or self._hierarchy_structure_ready
                ):
                    hierarchy_started = time.perf_counter()
                    rebuild = (
                        not self._hierarchy_structure_ready
                        or (
                            active_count >= minimum_active
                            and self._frame - self._hierarchy_structure_frame
                            >= self.config.amg_setup_interval_frames
                        )
                    )
                    if rebuild:
                        fine_matrix = self._assemble_cpu_matrix(gradients)
                        fine_diagonal = np.asarray(
                            fine_matrix.diagonal(), dtype=np.float64
                        )
                        if self.config.symmetric_diagonal_equilibration:
                            fine_scaling = 1.0 / np.sqrt(
                                np.maximum(fine_diagonal, 1.0e-30)
                            )
                            from scipy import sparse

                            scale_matrix = sparse.diags(
                                fine_scaling, format="csr"
                            )
                            fine_matrix = (
                                scale_matrix @ fine_matrix @ scale_matrix
                            ).tocsr()
                        self._refresh_levels(
                            fine_matrix, rebuild_structure=True
                        )
                    else:
                        fine_matrix_gpu = self._assemble_torch_system(
                            gradients, diagonal
                        )
                        self._refresh_levels_gpu(fine_matrix_gpu)
                    hierarchy_build_ms += (
                        time.perf_counter() - hierarchy_started
                    ) * 1000.0
                amg_ready_for_solve = bool(self._levels)
                rap_refreshed_for_current_matrix = (
                    self._rap_numeric_refreshes
                    > outer_rap_refreshes_before
                )
                (
                    delta_lambda,
                    iterations,
                    final_recursive_relative,
                    final_true_relative,
                    final_level0_operator_error,
                    residual_replacements,
                ) = self._pcg(rhs, gradients, diagonal)
                total_pcg += iterations
                correction = torch.zeros_like(current)
                weighted = delta_lambda[:, None, None] * gradients
                for local_index in range(4):
                    correction.index_add_(
                        0,
                        self.elements[:, local_index],
                        weighted[:, local_index],
                    )
                correction *= self.inverse_mass[:, None]
            if self.config.sqp_direction_enabled:
                direction_backend = "legacy_sqp"
                (
                    correction,
                    delta_lambda,
                    sqp_direction_metrics,
                ) = self._sqp_constrained_direction(
                    current,
                    material_residual,
                    constraints,
                    gradients,
                    correction,
                    delta_lambda,
                    diagonal,
                )
            maximum_norm = torch.max(torch.linalg.norm(correction, dim=1))
            global_scale = (
                torch.ones((), dtype=self.dtype, device=self.device)
                if self.config.maximum_correction_m is None
                else torch.clamp(
                    self.config.maximum_correction_m
                    / maximum_norm.clamp_min(1.0e-12),
                    max=1.0,
                )
            )
            # MGPBD is a global direction.  Per-vertex clipping destroys the
            # descent property; one scalar retains M^-1 J^T delta_lambda.
            correction *= global_scale
            relaxation = self.config.relaxation
            proposed_lagrangian = lagrangian + delta_lambda

            def objective_norm(values: object, multiplier: object) -> float:
                if self.config.line_search_objective == "dual":
                    values = values + self.alpha_tilde * multiplier
                return float(torch.linalg.norm(values))

            # Upstream updates the complete multiplier before searching only
            # the position step.  The downstream orientation-safe extension
            # instead treats position and multiplier as one atomic step.
            current_objective_multiplier = (
                lagrangian
                if self.config.line_search_scale_lagrangian
                else proposed_lagrangian
            )
            current_norm = objective_norm(
                constraints, current_objective_multiplier
            )
            constrained_direction_metrics = (
                soc_admm_direction_metrics
                if soc_admm_direction_metrics is not None
                else sqp_direction_metrics
            )
            constrained_merit_slope = (
                float(constrained_direction_metrics["merit_slope"])
                if constrained_direction_metrics is not None
                else None
            )
            if initial_dual_norm is None:
                initial_dual_norm = float(
                    torch.linalg.norm(constraints + self.alpha_tilde * lagrangian)
                )

            def evaluate_trial(step: float):
                trial_positions = current + step * correction
                trial_constraints, _trial_gradients, _trial_active = (
                    self.constraints_and_gradients(trial_positions)
                )
                _trial_policy, multiplier_step = lagrangian_acceptance_policy(
                    rejected=False,
                    rejected_policy=(
                        self.config.line_search_rejected_lagrangian_policy
                    ),
                    line_search_scale_lagrangian=(
                        self.config.line_search_scale_lagrangian
                    ),
                    accepted_step=step,
                    correction_global_scale=float(global_scale),
                )
                trial_multiplier = lagrangian + multiplier_step * delta_lambda
                trial_objective = objective_norm(
                    trial_constraints, trial_multiplier
                )
                trial_maximum_strain = float(torch.max(trial_constraints))
                trial_minimum_ratio = None
                if self.config.orientation_diagnostics_enabled:
                    trial_minimum_ratio = float(
                        torch.min(self.signed_volume_ratios(trial_positions))
                    )
                return (
                    trial_positions,
                    trial_constraints,
                    trial_multiplier,
                    trial_objective,
                    trial_minimum_ratio,
                    trial_maximum_strain,
                )

            (
                trial,
                trial_constraints,
                trial_lagrangian,
                trial_objective,
                trial_minimum_ratio,
                trial_maximum_strain,
            ) = evaluate_trial(relaxation)
            candidate_maximum_strain = trial_maximum_strain
            if self.config.orientation_diagnostics_enabled:
                orientation_record["candidate_minimum_signed_volume_ratio"] = float(
                    trial_minimum_ratio
                )
            outer_line_search_reductions = 0
            outer_orientation_backtracks = 0
            outer_strain_backtracks = 0
            minimum_step = (
                self.config.line_search_minimum_step
                if self.config.line_search_minimum_step is not None
                else self.config.relaxation / 4096.0
            )

            def objective_rejected(value: float, step: float) -> bool:
                if constrained_merit_slope is not None:
                    return armijo_merit_rejected(
                        value,
                        current_norm,
                        step,
                        constrained_merit_slope,
                        self.config.sqp_armijo_coefficient,
                    )
                return line_search_objective_rejected(
                    value,
                    current_norm,
                    self.config.line_search_acceptance_epsilon,
                )

            def orientation_rejected(value: float | None) -> bool:
                return bool(
                    self.config.orientation_guard_enabled
                    and (
                        value is None
                        or not np.isfinite(value)
                        or value < self.config.orientation_guard_minimum_ratio
                    )
                )

            def strain_rejected(value: float) -> bool:
                return strain_trust_filter_rejected(
                    value,
                    enabled=self.config.strain_trust_filter_enabled,
                    maximum=self.config.strain_trust_filter_maximum,
                )

            if self.config.line_search_enabled:
                while (
                    (
                        objective_rejected(trial_objective, relaxation)
                        or orientation_rejected(trial_minimum_ratio)
                        or strain_rejected(trial_maximum_strain)
                    )
                    and relaxation > minimum_step
                ):
                    if orientation_rejected(trial_minimum_ratio):
                        outer_orientation_backtracks += 1
                    if strain_rejected(trial_maximum_strain):
                        outer_strain_backtracks += 1
                    relaxation *= 0.5
                    line_search_reductions += 1
                    outer_line_search_reductions += 1
                    (
                        trial,
                        trial_constraints,
                        trial_lagrangian,
                        trial_objective,
                        trial_minimum_ratio,
                        trial_maximum_strain,
                    ) = evaluate_trial(relaxation)
            rejected = bool(
                self.config.line_search_enabled
                and (
                    objective_rejected(trial_objective, relaxation)
                    or orientation_rejected(trial_minimum_ratio)
                    or strain_rejected(trial_maximum_strain)
                    or relaxation < minimum_step
                )
            )
            multiplier_policy, lagrangian_update_fraction = (
                lagrangian_acceptance_policy(
                rejected=rejected,
                rejected_policy=(
                    self.config.line_search_rejected_lagrangian_policy
                ),
                line_search_scale_lagrangian=(
                    self.config.line_search_scale_lagrangian
                ),
                accepted_step=relaxation,
                correction_global_scale=float(global_scale),
                )
            )
            accepted_step = 0.0 if rejected else relaxation
            if rejected:
                trial = current
            # The same audited scalar drives the actual state transaction and
            # the artifact record.  This prevents a truthful label attached
            # to a different tensor update.
            accepted_lagrangian = (
                lagrangian + lagrangian_update_fraction * delta_lambda
            )
            current = trial
            lagrangian_update_norm = float(
                torch.linalg.norm(accepted_lagrangian - lagrangian)
            )
            delta_lambda_norm = float(torch.linalg.norm(delta_lambda))
            observed_lagrangian_update_norm_fraction = (
                lagrangian_update_norm / delta_lambda_norm
                if delta_lambda_norm > 1.0e-20
                else 0.0
            )
            expected_lagrangian = (
                lagrangian + lagrangian_update_fraction * delta_lambda
            )
            transaction_error_norm = float(
                torch.linalg.norm(accepted_lagrangian - expected_lagrangian)
            )
            transaction_relative_error = transaction_error_norm / max(
                delta_lambda_norm, 1.0e-20
            )
            if delta_lambda_norm <= 1.0e-20:
                lagrangian_fraction_matches_observed = True
            else:
                lagrangian_fraction_matches_observed = bool(
                    np.isclose(
                        observed_lagrangian_update_norm_fraction,
                        lagrangian_update_fraction,
                        rtol=1.0e-5,
                        atol=1.0e-7,
                    )
                )
            lagrangian = accepted_lagrangian
            if post_iteration is not None:
                current = post_iteration(current)
            accepted_constraints, _accepted_gradients, _accepted_active = (
                self.constraints_and_gradients(current)
            )
            self._last_accepted_outer_positions = current.detach().clone()
            self._last_accepted_outer_iteration = outer + 1
            dual_norm = float(
                torch.linalg.norm(accepted_constraints + self.alpha_tilde * lagrangian)
            )
            accepted_objective = objective_norm(
                accepted_constraints, lagrangian
            )
            sqp_direction_record: dict[str, object] | None = None
            soc_admm_direction_record: dict[str, object] | None = None
            if constrained_direction_metrics is not None:
                armijo_rhs = (
                    0.5 * current_norm * current_norm
                    + self.config.sqp_armijo_coefficient
                    * accepted_step
                    * float(constrained_merit_slope)
                )
                accepted_merit = 0.5 * accepted_objective * accepted_objective
                coupled_transaction = bool(
                    np.isclose(
                        lagrangian_update_fraction,
                        accepted_step,
                        rtol=1.0e-6,
                        atol=1.0e-7,
                    )
                    and lagrangian_fraction_matches_observed
                )
                constrained_direction_record = {
                    **constrained_direction_metrics,
                    "coupled_position_multiplier_transaction": (
                        coupled_transaction
                    ),
                    "accepted_step": float(accepted_step),
                    "accepted_multiplier_fraction": float(
                        lagrangian_update_fraction
                    ),
                    "multiplier_acceptance_policy": multiplier_policy,
                    "armijo_coefficient": float(
                        self.config.sqp_armijo_coefficient
                    ),
                    "armijo_merit_before": float(
                        0.5 * current_norm * current_norm
                    ),
                    "armijo_merit_after": float(accepted_merit),
                    "armijo_rhs": float(armijo_rhs),
                    "armijo_satisfied": bool(
                        not rejected and accepted_merit <= armijo_rhs
                    ),
                    "rolled_back_atomically": bool(
                        coupled_transaction
                        and (
                            not rejected
                            or (
                            accepted_step == 0.0
                            and lagrangian_update_fraction == 0.0
                            )
                        )
                    ),
                }
                if soc_admm_direction_metrics is not None:
                    soc_admm_direction_record = constrained_direction_record
                else:
                    sqp_direction_record = constrained_direction_record
            if self.config.orientation_diagnostics_enabled:
                accepted_volume_ratios = self.signed_volume_ratios(current)
                orientation_record.update(
                    {
                        "orientation_backtracks": int(
                            outer_orientation_backtracks
                        ),
                        "accepted_minimum_signed_volume_ratio": float(
                            torch.min(accepted_volume_ratios)
                        ),
                        "accepted_inverted_tetrahedra": int(
                            torch.count_nonzero(accepted_volume_ratios < 0.0)
                        ),
                        "accepted_collapsed_tetrahedra": int(
                            torch.count_nonzero(
                                accepted_volume_ratios <= 1.0e-8
                            )
                        ),
                        "position_and_lagrangian_step_accepted_atomically": bool(
                            not rejected
                            or multiplier_policy
                            == "rollback_multiplier_with_position"
                        ),
                        "lagrangian_update_l2": lagrangian_update_norm,
                        "delta_lambda_l2": delta_lambda_norm,
                        "lagrangian_update_fraction": (
                            lagrangian_update_fraction
                        ),
                        "lagrangian_update_norm_fraction": (
                            observed_lagrangian_update_norm_fraction
                        ),
                        "lagrangian_fraction_matches_observed": (
                            lagrangian_fraction_matches_observed
                        ),
                        "lagrangian_transaction_relative_error": (
                            transaction_relative_error
                        ),
                        "lagrangian_acceptance_policy": multiplier_policy,
                        "accepted_full_delta_lambda": bool(
                            not rejected
                            and multiplier_policy
                            == "accepted_full_trial_multiplier"
                        ),
                        "rejected_retained_full_delta_lambda": bool(
                            rejected
                            and multiplier_policy
                            == "retain_full_public_multiplier"
                        ),
                        "rejected_rolled_back_delta_lambda": bool(
                            rejected
                            and multiplier_policy
                            == "rollback_multiplier_with_position"
                        ),
                        "lagrangian_rolled_back_with_rejected_position": bool(
                            not rejected or lagrangian_update_norm == 0.0
                        ),
                    }
                )
            outer_history.append(
                {
                    "outer_iteration": outer + 1,
                    "direction_backend": direction_backend,
                    "legacy_direction_pcg_skipped": bool(
                        legacy_direction_pcg_skipped
                    ),
                    "active_constraints": active_count,
                    "amg_ready_for_solve": amg_ready_for_solve,
                    "rap_refreshed_for_current_matrix": rap_refreshed_for_current_matrix,
                    "rap_numeric_refreshes_total": self._rap_numeric_refreshes,
                    "fine_diagonal_sum": fine_diagonal_sum,
                    "fine_diagonal_l2": fine_diagonal_l2,
                    "pcg_iterations": int(iterations),
                    "pcg_relative_residual": float(final_true_relative),
                    "pcg_true_relative_residual": float(final_true_relative),
                    "pcg_recursive_relative_residual": float(
                        final_recursive_relative
                    ),
                    "pcg_true_residual_replacements": int(
                        residual_replacements
                    ),
                    "level0_physical_operator_relative_error": float(
                        final_level0_operator_error
                    ),
                    "line_search_step": float(accepted_step),
                    "line_search_reductions": int(outer_line_search_reductions),
                    "line_search_rejected": rejected,
                    "line_search_objective_before": float(current_norm),
                    "line_search_objective_after": float(accepted_objective),
                    "correction_maximum_before_clip": float(maximum_norm),
                    "correction_global_scale": float(global_scale),
                    "arap_l2_before": float(torch.linalg.norm(constraints)),
                    "arap_l2_after": float(torch.linalg.norm(accepted_constraints)),
                    "arap_maximum_before": float(torch.max(constraints)),
                    "arap_maximum_after": float(
                        torch.max(accepted_constraints)
                    ),
                    "strain_trust_filter_is_downstream_extension": bool(
                        self.config.strain_trust_filter_enabled
                    ),
                    "strain_trust_filter_maximum": float(
                        self.config.strain_trust_filter_maximum
                    ),
                    "candidate_maximum_arap_strain": float(
                        candidate_maximum_strain
                    ),
                    "last_trial_maximum_arap_strain": float(
                        trial_maximum_strain
                    ),
                    "accepted_maximum_arap_strain": float(
                        torch.max(accepted_constraints)
                    ),
                    "strain_trust_backtracks": int(outer_strain_backtracks),
                    "dual_l2_after": dual_norm,
                    **(
                        {"sqp_direction": sqp_direction_record}
                        if sqp_direction_record is not None
                        else {}
                    ),
                    **(
                        {"soc_admm_direction": soc_admm_direction_record}
                        if soc_admm_direction_record is not None
                        else {}
                    ),
                    **orientation_record,
                }
            )
            absolute_met = (
                self.config.outer_absolute_residual is not None
                and dual_norm <= self.config.outer_absolute_residual
            )
            relative_met = (
                self.config.outer_relative_residual is not None
                and initial_dual_norm is not None
                and dual_norm <= self.config.outer_relative_residual * initial_dual_norm
            )
            if absolute_met or relative_met:
                break
        final_constraints, _gradients, _active = self.constraints_and_gradients(current)
        self._frame += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_metrics = {
            "available": True,
            "projection_failed": False,
            "projected_frames": self._frame,
            "constraint_kind": "one_ARAP_singular_value_norm_per_tetrahedron",
            "direction_backend": (
                "soc_admm"
                if self.config.soc_admm_direction_enabled
                else (
                    "legacy_sqp"
                    if self.config.sqp_direction_enabled
                    else "mgpbd_pcg"
                )
            ),
            "legacy_direction_pcg_skipped": bool(
                self.config.soc_admm_direction_enabled
                and outer_history
                and all(
                    bool(record["legacy_direction_pcg_skipped"])
                    for record in outer_history
                )
            ),
            "constraints": int(len(self.elements_np)),
            "amg_enabled": bool(self._levels),
            "amg_level_sizes": [int(level.matrix.shape[0]) for level in self._levels],
            "amg_structure_builds": self._structure_builds,
            "amg_rap_numeric_refreshes_total": self._rap_numeric_refreshes,
            "amg_rap_numeric_refreshes_last_project": (
                self._rap_numeric_refreshes - project_rap_refreshes_before
            ),
            "all_amg_outer_matrices_refreshed": all(
                not bool(record["amg_ready_for_solve"])
                or bool(record["rap_refreshed_for_current_matrix"])
                for record in outer_history
            ),
            "amg_hierarchy_builder": self._hierarchy_builder,
            "amg_refresh_ms_last": hierarchy_build_ms,
            "pcg_iterations_total": int(total_pcg),
            "soc_admm_iterations_total": int(total_soc_admm_iterations),
            "soc_admm_pcg_iterations_total": int(total_soc_admm_pcg),
            "pcg_relative_residual_final": float(final_true_relative),
            "pcg_true_relative_residual_final": float(final_true_relative),
            "pcg_recursive_relative_residual_final": float(
                final_recursive_relative
            ),
            "level0_physical_operator_relative_error_final": float(
                final_level0_operator_error
            ),
            "arap_constraint_max_before": initial_max,
            "arap_constraint_max_after": float(torch.max(final_constraints)),
            "arap_constraint_p95_after": float(torch.quantile(final_constraints, 0.95)),
            "line_search_reductions": int(line_search_reductions),
            "symmetric_diagonal_equilibration": self.config.symmetric_diagonal_equilibration,
            "smoother_weight_runtime": float(self._runtime_smoother_weight),
            "unscaled_diagonal_dynamic_range": float(diagonal_dynamic_range),
            "mass_model": self.mass_model,
            "nonlinear_iterations_completed": len(outer_history),
            "outer_dual_initial": float(initial_dual_norm or 0.0),
            "outer_dual_final": float(
                outer_history[-1]["dual_l2_after"] if outer_history else 0.0
            ),
            "outer_iterations": outer_history,
            "project_ms_last": elapsed_ms,
            "configuration": self.config.to_dict(),
        }
        return current


@dataclass(frozen=True)
class MGPBDTetTopology:
    """Local rest tet mesh plus its boundary shell for contact diagnostics."""

    rest_positions: np.ndarray
    elements: np.ndarray
    boundary_faces: np.ndarray
    boundary_indices: np.ndarray

    @property
    def shell_vertices(self) -> np.ndarray:
        return self.rest_positions[self.boundary_indices]


class MGPBDTetSolver:
    """Own MGPBD time integration and paper-style post-projection contact."""

    def __init__(
        self,
        rest_positions_local: np.ndarray,
        elements: np.ndarray,
        config: MGPBDTetConfig,
        *,
        initial_center_m: tuple[float, float, float],
        total_mass_kg: float,
        table_height_m: float,
        device: object,
        tetrahedralization: dict[str, object] | None = None,
    ):
        import torch

        config.validate()
        if config.maximum_correction_m is None:
            raise ValueError(
                "contact-enabled MGPBD solver requires a finite correction limit"
            )
        rest_local = np.asarray(rest_positions_local, dtype=np.float32).reshape(-1, 3)
        elements = np.asarray(elements, dtype=np.int32).reshape(-1, 4)
        faces = boundary_faces(elements)
        boundary_indices = np.unique(faces).astype(np.int32)
        self.topology = MGPBDTetTopology(rest_local, elements, faces, boundary_indices)
        self.config = config
        self.device = device
        self.dtype = torch.float32
        self.table_height_m = float(table_height_m)
        self.total_mass_kg = float(total_mass_kg)
        self._origin = np.asarray(initial_center_m, dtype=np.float32).reshape(3)
        world_rest = torch.as_tensor(
            rest_local + self._origin, dtype=self.dtype, device=device
        )
        self.projector = VolumetricMGPBDProjector(
            world_rest,
            elements,
            total_mass_kg=total_mass_kg,
            config=config,
        )
        self.positions = world_rest.clone()
        self.velocities = torch.zeros_like(self.positions)
        self._boundary_indices = torch.as_tensor(
            boundary_indices, dtype=torch.long, device=device
        )
        collision_sample_faces, collision_sample_weights = triangle_contact_samples(
            faces
        )
        self._collision_sample_faces = torch.as_tensor(
            collision_sample_faces, dtype=torch.long, device=device
        )
        self._collision_sample_weights = torch.as_tensor(
            collision_sample_weights, dtype=self.dtype, device=device
        )
        rest_tets = rest_local[elements]
        self._rest_signed_six_volume = np.einsum(
            "ij,ij->i",
            np.cross(
                rest_tets[:, 1] - rest_tets[:, 0],
                rest_tets[:, 2] - rest_tets[:, 0],
            ),
            rest_tets[:, 3] - rest_tets[:, 0],
        )
        self._rest_signed_six_volume_tensor = torch.as_tensor(
            self._rest_signed_six_volume,
            dtype=self.dtype,
            device=device,
        )
        self._collider_centers = torch.empty((0, 3), dtype=self.dtype, device=device)
        self._collider_rotations = torch.empty((0, 3, 3), dtype=self.dtype, device=device)
        self._previous_collider_centers = torch.empty(
            (0, 3), dtype=self.dtype, device=device
        )
        self._previous_collider_rotations = torch.empty(
            (0, 3, 3), dtype=self.dtype, device=device
        )
        self._collider_reference_points = torch.empty(
            (0, 3), dtype=self.dtype, device=device
        )
        self._previous_collider_reference_points = torch.empty(
            (0, 3), dtype=self.dtype, device=device
        )
        self._plane_normals = torch.empty((0, 0, 3), dtype=self.dtype, device=device)
        self._plane_offsets = torch.empty((0, 0), dtype=self.dtype, device=device)
        self._plane_counts = np.empty(0, dtype=np.int32)
        self._friction_scales = torch.empty(0, dtype=self.dtype, device=device)
        self._collider_arm_indices = np.empty(0, dtype=np.int32)
        self._collider_moving_mask = np.empty(0, dtype=bool)
        self._frame_contact_counts = np.empty(0, dtype=np.int32)
        self._contact_count_peak = np.empty(0, dtype=np.int32)
        self._frame_sample_contact_counts = np.empty(0, dtype=np.int32)
        self._sample_contact_count_peak = np.empty(0, dtype=np.int32)
        self._contact_active_by_set: dict[str, object] = {}
        self._contact_anchor_local_by_set: dict[str, object] = {}
        self._contact_preload_by_set: dict[str, object] = {}
        self._frame_maximum_normal_sweep_m = 0.0
        self._peak_maximum_normal_sweep_m = 0.0
        self._grasp_active_by_arm = np.empty(0, dtype=bool)
        self._grasp_miss_frames_by_arm = np.empty(0, dtype=np.int32)
        self._coarse_grasp_transport_frames = 0
        self._coarse_grasp_transport_peak_m = 0.0
        self._coarse_grasp_transport_cumulative_m = np.zeros(3, dtype=np.float64)
        self._coarse_grasp_center_lock_frames = 0
        self._coarse_grasp_center_lock_peak_m = 0.0
        self._coarse_grasp_center_lock_cumulative_m = np.zeros(
            3, dtype=np.float64
        )
        self._volume_barrier_activations = 0
        self._volume_barrier_minimum_step_fraction = 1.0
        self._volume_barrier_worst_candidate_ratio = 1.0
        self._table_contacts = 0
        self._table_contacts_peak = 0
        self._visual_contact_projection: dict[str, object] = {
            "enabled": False,
            "passes": 3,
            "clearance_m": 0.00025,
            "projected_vertices": 0,
            "maximum_pre_projection_penetration_m": 0.0,
            "maximum_post_projection_penetration_m": 0.0,
        }
        self._tetrahedralization = dict(tetrahedralization or {})
        self._last_step_ms = 0.0

    def reset(self, origin: np.ndarray, rotation: np.ndarray | None = None) -> None:
        import torch

        origin = np.asarray(origin, dtype=np.float32).reshape(3)
        if rotation is None:
            rotation = np.eye(3, dtype=np.float32)
        rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        world = self.topology.rest_positions @ rotation + origin
        self.positions = torch.as_tensor(world, dtype=self.dtype, device=self.device)
        self.velocities.zero_()
        self.projector.rest_positions = self.positions.detach().clone()
        self._origin = origin
        for values in self._contact_active_by_set.values():
            values.zero_()
        for values in self._contact_anchor_local_by_set.values():
            values.zero_()
        for values in self._contact_preload_by_set.values():
            values.zero_()
        self._grasp_active_by_arm.fill(False)
        self._grasp_miss_frames_by_arm.fill(0)
        if len(self._collider_centers):
            self._previous_collider_centers = self._collider_centers.clone()
            self._previous_collider_rotations = self._collider_rotations.clone()
        if len(self._collider_reference_points):
            self._previous_collider_reference_points = (
                self._collider_reference_points.clone()
            )

    def zero_velocity(self) -> None:
        self.velocities.zero_()

    def shell_positions(self) -> np.ndarray:
        return (
            self.positions[self._boundary_indices].detach().cpu().numpy().astype(np.float32)
        )

    def set_convex_colliders(
        self,
        centers: np.ndarray,
        rotations: np.ndarray,
        plane_normals: np.ndarray,
        plane_offsets: np.ndarray,
        plane_counts: np.ndarray,
    ) -> None:
        import torch

        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        rotations = np.asarray(rotations, dtype=np.float32).reshape(-1, 3, 3)
        plane_normals = np.asarray(plane_normals, dtype=np.float32)
        plane_offsets = np.asarray(plane_offsets, dtype=np.float32)
        plane_counts = np.asarray(plane_counts, dtype=np.int32).reshape(-1)
        count = len(centers)
        if rotations.shape[0] != count or plane_normals.shape[0] != count:
            raise ValueError("MGPBD collider array count mismatch")
        new_centers = torch.as_tensor(
            centers, dtype=self.dtype, device=self.device
        )
        new_rotations = torch.as_tensor(
            rotations, dtype=self.dtype, device=self.device
        )
        if len(self._collider_centers) == count:
            self._previous_collider_centers = self._collider_centers.clone()
            self._previous_collider_rotations = self._collider_rotations.clone()
        else:
            self._previous_collider_centers = new_centers.clone()
            self._previous_collider_rotations = new_rotations.clone()
        self._collider_centers = new_centers
        self._collider_rotations = new_rotations
        self._plane_normals = torch.as_tensor(plane_normals, dtype=self.dtype, device=self.device)
        self._plane_offsets = torch.as_tensor(plane_offsets, dtype=self.dtype, device=self.device)
        self._plane_counts = plane_counts
        if len(self._friction_scales) != count:
            self._friction_scales = torch.ones(count, dtype=self.dtype, device=self.device)
            self._frame_contact_counts = np.zeros(count, dtype=np.int32)
            self._contact_count_peak = np.zeros(count, dtype=np.int32)
            self._frame_sample_contact_counts = np.zeros(count, dtype=np.int32)
            self._sample_contact_count_peak = np.zeros(count, dtype=np.int32)
            if count % 3 == 0:
                arm_indices = np.repeat(
                    np.arange(count // 3, dtype=np.int32), 3
                )
                moving_mask = np.tile((False, False, True), count // 3)
            elif count % 2 == 0:
                arm_indices = np.repeat(
                    np.arange(count // 2, dtype=np.int32), 2
                )
                moving_mask = np.tile((False, True), count // 2)
            else:
                raise ValueError("MGPBD gripper colliders cannot be grouped by arm")
            self.set_collider_groups(arm_indices, moving_mask)
            self._contact_active_by_set.clear()
            self._contact_anchor_local_by_set.clear()
            self._contact_preload_by_set.clear()
        for name, point_count in (
            ("vertices", len(self._boundary_indices)),
            ("samples", len(self._collision_sample_faces)),
        ):
            shape = (point_count, count)
            active = self._contact_active_by_set.get(name)
            if active is None or tuple(active.shape) != shape:
                self._contact_active_by_set[name] = torch.zeros(
                    shape, dtype=torch.bool, device=self.device
                )
                self._contact_anchor_local_by_set[name] = torch.zeros(
                    (*shape, 3), dtype=self.dtype, device=self.device
                )
                self._contact_preload_by_set[name] = torch.zeros(
                    shape, dtype=self.dtype, device=self.device
                )

    def set_collider_groups(
        self, arm_indices: np.ndarray, moving_mask: np.ndarray
    ) -> None:
        """Declare arm ownership and fixed/moving roles for contact proxies."""

        arm_indices = np.asarray(arm_indices, dtype=np.int32).reshape(-1)
        moving_mask = np.asarray(moving_mask, dtype=bool).reshape(-1)
        count = len(self._collider_centers)
        if len(arm_indices) != count or len(moving_mask) != count or count == 0:
            raise ValueError("MGPBD collider group metadata count mismatch")
        if int(arm_indices.min()) < 0:
            raise ValueError("MGPBD collider arm indices must be nonnegative")
        arm_count = int(arm_indices.max()) + 1
        if set(arm_indices.tolist()) != set(range(arm_count)):
            raise ValueError("MGPBD collider arm indices must be contiguous")
        for arm_index in range(arm_count):
            owned = arm_indices == arm_index
            if not np.any(owned & ~moving_mask) or not np.any(owned & moving_mask):
                raise ValueError("each MGPBD arm requires fixed and moving proxies")
        self._collider_arm_indices = arm_indices.copy()
        self._collider_moving_mask = moving_mask.copy()
        self._grasp_active_by_arm = np.zeros(arm_count, dtype=bool)
        self._grasp_miss_frames_by_arm = np.zeros(arm_count, dtype=np.int32)

    def set_collider_reference_points(self, points: np.ndarray) -> None:
        """Track contact-proxy centroids for common gripper translation."""

        import torch

        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points) != len(self._collider_centers):
            raise ValueError("MGPBD collider reference point count mismatch")
        current = torch.as_tensor(
            points, dtype=self.dtype, device=self.device
        )
        if len(self._collider_reference_points) == len(current):
            self._previous_collider_reference_points = (
                self._collider_reference_points.clone()
            )
        else:
            self._previous_collider_reference_points = current.clone()
        self._collider_reference_points = current

    def set_collider_friction_scales(self, scales: np.ndarray) -> None:
        import torch

        scales = np.asarray(scales, dtype=np.float32).reshape(-1)
        if len(scales) != len(self._collider_centers):
            raise ValueError("MGPBD collider friction scale count mismatch")
        self._friction_scales = torch.as_tensor(scales, dtype=self.dtype, device=self.device)

    def _project_convex_points(
        self,
        points: object,
        previous_points: object,
        *,
        contact_set: str,
    ):
        import torch

        radius = self.config.particle_radius_m
        active_state_all = self._contact_active_by_set[contact_set]
        anchor_local_all = self._contact_anchor_local_by_set[contact_set]
        preload_all = self._contact_preload_by_set[contact_set]
        counts = np.zeros(len(self._collider_centers), dtype=np.int32)
        for collider_index in range(len(self._collider_centers)):
            count = int(self._plane_counts[collider_index])
            if count <= 0:
                continue
            rotation = self._collider_rotations[collider_index]
            local = (points - self._collider_centers[collider_index]) @ rotation
            normals = self._plane_normals[collider_index, :count]
            offsets = self._plane_offsets[collider_index, :count]
            plane_distance = local @ normals.T - offsets[None, :]
            signed, plane_index = torch.max(plane_distance, dim=1)
            local_normal = normals[plane_index]
            world_normal = local_normal @ rotation.T

            # Only the collider motion along the active outward normal may
            # enlarge the contact skin.  The old scalar norm also inflated a
            # jaw during tangential lift, producing a 15 mm invisible balloon
            # that pushed the toy away instead of carrying it.
            previous_world = (
                local
                @ self._previous_collider_rotations[collider_index].T
                + self._previous_collider_centers[collider_index]
            )
            surface_motion = points - previous_world
            normal_sweep = torch.clamp(
                torch.sum(surface_motion * world_normal, dim=1),
                min=0.0,
                max=self.config.maximum_sweep_margin_m,
            )
            if len(normal_sweep):
                maximum_sweep = float(torch.max(normal_sweep))
                self._frame_maximum_normal_sweep_m = max(
                    self._frame_maximum_normal_sweep_m, maximum_sweep
                )
                self._peak_maximum_normal_sweep_m = max(
                    self._peak_maximum_normal_sweep_m, maximum_sweep
                )

            surface_distance = signed - radius
            prior_active = active_state_all[:, collider_index].clone()
            near = surface_distance < self.config.contact_release_m
            touching = surface_distance < (
                self.config.contact_slop_m + normal_sweep
            )
            active = near & (touching | prior_active)
            counts[collider_index] = int(torch.count_nonzero(active))
            active_state_all[:, collider_index] = active
            preload = preload_all[:, collider_index]
            preload[~active] = 0.0
            if not bool(torch.any(active)):
                continue

            anchor_local = anchor_local_all[:, collider_index]
            newly_active = active & ~prior_active
            anchor_local[newly_active] = local[newly_active]
            penetration = torch.clamp(
                radius + normal_sweep - signed,
                min=0.0,
                max=self.config.maximum_correction_m,
            )
            points += torch.where(
                active[:, None],
                penetration[:, None] * world_normal,
                torch.zeros_like(points),
            )

            # Persistent local anchors implement static friction in the
            # moving collider frame.  World-relative friction pinned the toy
            # to the table while the jaw moved, so it could be pushed but not
            # lifted.  The anchor follows both jaw translation and rotation;
            # Coulomb-limited slip refreshes it rather than welding the toy.
            preload[:] = torch.where(
                active,
                torch.clamp(
                    torch.maximum(
                        torch.full_like(preload, self.config.contact_slop_m),
                        preload * 0.999 + penetration,
                    ),
                    max=self.config.contact_release_m,
                ),
                torch.zeros_like(preload),
            )
            anchor_world = anchor_local @ rotation.T + self._collider_centers[
                collider_index
            ]
            anchor_error = anchor_world - points
            normal_error = torch.sum(anchor_error * world_normal, dim=1)
            tangent = anchor_error - normal_error[:, None] * world_normal
            tangent_norm = torch.linalg.norm(tangent, dim=1)
            friction_limit = (
                self.config.gripper_friction
                * float(self._friction_scales[collider_index])
                * preload
            )
            friction_scale = torch.minimum(
                torch.ones_like(tangent_norm),
                friction_limit / tangent_norm.clamp_min(1.0e-12),
            )
            points += torch.where(
                active[:, None],
                tangent * friction_scale[:, None],
                torch.zeros_like(points),
            )
            slipping = active & (tangent_norm > 1.0e-8) & (friction_scale < 0.999)
            if bool(torch.any(slipping)):
                current_local = (
                    points - self._collider_centers[collider_index]
                ) @ rotation
                anchor_local[slipping] = current_local[slipping]
        return points, counts

    def _project_collisions(self, positions: object, previous: object):
        import torch

        boundary = self._boundary_indices
        points = positions[boundary]
        previous_points = previous[boundary]
        radius = self.config.particle_radius_m
        table_penetration = self.table_height_m + radius - points[:, 2]
        table_active = table_penetration > 0.0
        if bool(torch.any(table_active)):
            # A horizontal support plane acts first on the rigid translation
            # null mode.  Projecting each fine boundary node independently
            # folded the first layer of 4 mm tetrahedra, after which the
            # positive-volume barrier rejected almost the entire support
            # correction.  Lift the complete body by the deepest penetration
            # instead: this resolves every table contact while preserving all
            # signed volumes exactly.  Deformation remains owned by MGPBD and
            # by the non-uniform gripper contacts below.
            correction = torch.clamp(
                torch.max(table_penetration),
                min=0.0,
                max=self.config.maximum_correction_m,
            )
            positions[:, 2] += correction

            # Apply support friction through the same rigid null mode.  A
            # per-node tangential clamp would reintroduce artificial shear at
            # the bottom surface before the constitutive solve sees it.
            points = positions[boundary]
            mean_tangent = torch.mean(
                points[table_active, :2] - previous_points[table_active, :2],
                dim=0,
            )
            tangent_norm = torch.linalg.norm(mean_tangent)
            removable = torch.minimum(
                tangent_norm,
                self.config.table_friction * correction,
            )
            positions[:, :2] -= mean_tangent * (
                removable / tangent_norm.clamp_min(1.0e-12)
            )
            points = positions[boundary]
        table_count = int(torch.count_nonzero(table_active))
        points, vertex_counts = self._project_convex_points(
            points, previous_points, contact_set="vertices"
        )
        positions[boundary] = points

        # Close the gaps between boundary nodes by sampling every triangle at
        # its centroid and edge midpoints. A sample displacement
        # is scattered back to the existing vertices, so the MGPBD system and
        # its one-constraint-per-tet formulation remain unchanged.
        faces = self._collision_sample_faces
        weights = self._collision_sample_weights
        samples = torch.sum(positions[faces] * weights[:, :, None], dim=1)
        previous_samples = torch.sum(
            previous[faces] * weights[:, :, None], dim=1
        )
        corrected_samples, sample_counts = self._project_convex_points(
            samples.clone(), previous_samples, contact_set="samples"
        )
        sample_delta = corrected_samples - samples
        active_samples = torch.linalg.norm(sample_delta, dim=1) > 1.0e-9
        if bool(torch.any(active_samples)):
            accumulated = torch.zeros_like(positions)
            contributions = torch.zeros(
                (len(positions), 1), dtype=self.dtype, device=self.device
            )
            for local_index in range(3):
                supported = active_samples & (weights[:, local_index] > 0.0)
                if bool(torch.any(supported)):
                    indices = faces[supported, local_index]
                    accumulated.index_add_(0, indices, sample_delta[supported])
                    contributions.index_add_(
                        0,
                        indices,
                        torch.ones(
                            (int(torch.count_nonzero(supported)), 1),
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    )
            positions += accumulated / contributions.clamp_min(1.0)

        counts = vertex_counts + sample_counts
        active_grasps = []
        for arm_index in range(len(self._grasp_active_by_arm)):
            owned = self._collider_arm_indices == arm_index
            fixed = owned & ~self._collider_moving_mask
            moving = owned & self._collider_moving_mask
            moving_indices = np.flatnonzero(moving)
            closure = max(
                float(self._friction_scales[index])
                for index in moving_indices.tolist()
            )
            active_grasps.append(
                closure >= self.config.two_finger_transfer_closure_threshold
                and int(np.sum(counts[fixed])) > 0
                and int(np.sum(counts[moving])) > 0
            )
        if len(active_grasps) == len(self._grasp_active_by_arm):
            for arm_index, raw_active in enumerate(active_grasps):
                moving_indices = np.flatnonzero(
                    (self._collider_arm_indices == arm_index)
                    & self._collider_moving_mask
                )
                closure = max(
                    float(self._friction_scales[index])
                    for index in moving_indices.tolist()
                )
                if closure < self.config.two_finger_transfer_closure_threshold:
                    self._grasp_active_by_arm[arm_index] = False
                    self._grasp_miss_frames_by_arm[arm_index] = 0
                elif raw_active:
                    self._grasp_active_by_arm[arm_index] = True
                    self._grasp_miss_frames_by_arm[arm_index] = 0
                elif (
                    self._grasp_active_by_arm[arm_index]
                    and self._grasp_miss_frames_by_arm[arm_index]
                    < self.config.grasp_contact_persistence_frames
                ):
                    self._grasp_miss_frames_by_arm[arm_index] += 1
                else:
                    self._grasp_active_by_arm[arm_index] = False
        self._table_contacts = table_count
        self._table_contacts_peak = max(self._table_contacts_peak, table_count)
        self._frame_contact_counts = counts
        self._frame_sample_contact_counts = sample_counts
        if len(counts):
            self._contact_count_peak = np.maximum(self._contact_count_peak, counts)
            self._sample_contact_count_peak = np.maximum(
                self._sample_contact_count_peak, sample_counts
            )
        return self._enforce_volume_orientation(positions, previous)

    def _signed_volume_ratios(self, positions: object):
        import torch

        tets = positions[self.projector.elements]
        signed_six = torch.sum(
            torch.cross(
                tets[:, 1] - tets[:, 0],
                tets[:, 2] - tets[:, 0],
                dim=1,
            )
            * (tets[:, 3] - tets[:, 0]),
            dim=1,
        )
        return signed_six / self._rest_signed_six_volume_tensor

    def _enforce_volume_orientation(self, candidate: object, previous: object):
        """Line-search contact motion against a positive-volume barrier."""

        import torch

        minimum = self.config.minimum_signed_volume_ratio
        if minimum <= 0.0:
            return candidate
        candidate_ratio = float(torch.min(self._signed_volume_ratios(candidate)))
        self._volume_barrier_worst_candidate_ratio = min(
            self._volume_barrier_worst_candidate_ratio, candidate_ratio
        )
        if candidate_ratio >= minimum:
            return candidate
        # Translation is an exact ARAP null mode.  Move the prior valid shape
        # to the candidate center before line-searching deformation, so the
        # barrier never cancels a legitimate grasp lift merely to avoid a
        # local tet inversion.
        safe_previous = previous + (
            torch.mean(candidate, dim=0) - torch.mean(previous, dim=0)
        )
        previous_ratio = float(
            torch.min(self._signed_volume_ratios(safe_previous))
        )
        if previous_ratio < minimum:
            # This path is only reachable if an invalid state predates the
            # barrier.  Preserve the less-degenerate endpoint and let the ARAP
            # solve recover instead of accepting a still-worse contact step.
            return (
                safe_previous
                if previous_ratio >= candidate_ratio
                else candidate
            )
        delta = candidate - safe_previous
        low = 0.0
        high = 1.0
        for _ in range(12):
            middle = 0.5 * (low + high)
            trial = safe_previous + middle * delta
            if float(torch.min(self._signed_volume_ratios(trial))) >= minimum:
                low = middle
            else:
                high = middle
        self._volume_barrier_activations += 1
        self._volume_barrier_minimum_step_fraction = min(
            self._volume_barrier_minimum_step_fraction, low
        )
        return safe_previous + low * delta

    def _transport_active_grasps(self, positions: object):
        """Transport the MGPBD rigid null mode with a closed two-finger grasp."""

        import torch

        translations = []
        for arm_index, active in enumerate(self._grasp_active_by_arm.tolist()):
            if not active:
                continue
            # A fixed-finger proxy carries the common arm motion while
            # excluding the moving jaw's differential closure motion.
            fixed_indices = np.flatnonzero(
                (self._collider_arm_indices == arm_index)
                & ~self._collider_moving_mask
            )
            if not len(fixed_indices):
                continue
            indices = torch.as_tensor(
                fixed_indices, dtype=torch.long, device=self.device
            )
            if len(self._collider_reference_points) == len(
                self._collider_centers
            ):
                translation = torch.mean(
                    self._collider_reference_points[indices]
                    - self._previous_collider_reference_points[indices],
                    dim=0,
                )
            else:
                translation = torch.mean(
                    self._collider_centers[indices]
                    - self._previous_collider_centers[indices],
                    dim=0,
                )
            translations.append(translation)
        if not translations:
            return positions, torch.zeros_like(positions)
        displacement = torch.mean(torch.stack(translations, dim=0), dim=0)
        maximum = torch.linalg.norm(displacement)
        scale = torch.clamp(
            self.config.maximum_grasp_transport_m
            / maximum.clamp_min(1.0e-12),
            max=1.0,
        )
        result = (
            positions
            + self.config.two_finger_coarse_transfer_gain * scale * displacement[None]
        )
        self._coarse_grasp_transport_frames += 1
        applied = (result[0] - positions[0]).detach().cpu().numpy()
        self._coarse_grasp_transport_cumulative_m += applied
        self._coarse_grasp_transport_peak_m = max(
            self._coarse_grasp_transport_peak_m,
            float(torch.linalg.norm(result[0] - positions[0])),
        )
        return result, result - positions

    def step(self, *, synchronize: bool = False) -> None:
        import torch

        started = time.perf_counter()
        self._frame_maximum_normal_sweep_m = 0.0
        previous = self.positions.detach().clone()
        self.velocities *= self.config.damping_retention
        self.velocities[:, 2] -= 9.81 * self.config.dt_s
        transported, grasp_transport = self._transport_active_grasps(
            self.positions
        )
        predicted = transported + self.config.dt_s * self.velocities

        def collide(candidate: object):
            result = candidate
            for _ in range(self.config.collision_passes):
                result = self._project_collisions(result, previous)
            return result

        self.positions = self.projector.project(predicted, post_iteration=collide)
        center_lock = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.positions = collide(self.positions)
        common_center_motion = torch.mean(grasp_transport, dim=0)
        if bool(np.any(self._grasp_active_by_arm)):
            actual_center_motion = torch.mean(
                self.positions - previous, dim=0
            )
            # Pinch friction must support gravity even when the gripper is
            # stationary.  Restore any missing upward/common Z motion as a
            # rigid null-mode translation; never force the body downward into
            # the table.  X/Y closure deformation remains free.
            vertical = torch.clamp(
                common_center_motion[2] - actual_center_motion[2],
                min=0.0,
                max=self.config.maximum_grasp_transport_m,
            )
            center_lock[2] = vertical

            horizontal = common_center_motion[:2]
            horizontal_norm = torch.linalg.norm(horizontal)
            if float(horizontal_norm) > 1.0e-9:
                direction = horizontal / horizontal_norm
                delivered = torch.dot(actual_center_motion[:2], direction)
                missing = torch.clamp(
                    horizontal_norm - delivered,
                    min=0.0,
                    max=horizontal_norm,
                )
                center_lock[:2] = missing * direction
            lock_norm = torch.linalg.norm(center_lock)
            if float(lock_norm) > 1.0e-9:
                lock_scale = torch.clamp(
                    self.config.maximum_grasp_transport_m / lock_norm,
                    max=1.0,
                )
                center_lock *= lock_scale
                self.positions += center_lock
                self._coarse_grasp_center_lock_frames += 1
                self._coarse_grasp_center_lock_cumulative_m += (
                    center_lock.detach().cpu().numpy()
                )
                self._coarse_grasp_center_lock_peak_m = max(
                    self._coarse_grasp_center_lock_peak_m,
                    float(torch.linalg.norm(center_lock)),
                )
        # The measured jaw transport is prescribed motion and must not be
        # extrapolated as material velocity.  The contact/friction correction
        # is different: like a standard PBD collision impulse it must remain
        # in the updated velocity so it cancels downward slip on the next
        # frame.  Subtracting it recreated the same 12 mm fall every step.
        self.velocities = (
            self.positions - previous - grasp_transport
        ) / self.config.dt_s
        if synchronize and self.positions.is_cuda:
            torch.cuda.synchronize()
        self._last_step_ms = (time.perf_counter() - started) * 1000.0

    def signed_distances(self, points: np.ndarray, *, radius_m: float = 0.0) -> np.ndarray:
        import torch

        points_t = torch.as_tensor(
            np.asarray(points, dtype=np.float32).reshape(-1, 3),
            dtype=self.dtype,
            device=self.device,
        )
        output = torch.empty(
            (len(points_t), len(self._collider_centers)),
            dtype=self.dtype,
            device=self.device,
        )
        for collider_index in range(len(self._collider_centers)):
            count = int(self._plane_counts[collider_index])
            rotation = self._collider_rotations[collider_index]
            local = (points_t - self._collider_centers[collider_index]) @ rotation
            distance = (
                local @ self._plane_normals[collider_index, :count].T
                - self._plane_offsets[collider_index, :count][None, :]
            )
            output[:, collider_index] = torch.max(distance, dim=1).values - radius_m
        return output.detach().cpu().numpy()

    def contact_influence_by_node(self, *, band_m: float = 0.012):
        """Return a smooth per-node proximity field for visual contact skinning."""

        import torch

        if band_m <= 0.0:
            raise ValueError("contact influence band must be positive")
        with torch.no_grad():
            if not len(self._collider_centers):
                return torch.zeros(
                    len(self.positions), dtype=self.dtype, device=self.device
                )
            faces = self._collision_sample_faces
            samples = torch.sum(
                self.positions[faces]
                * self._collision_sample_weights[:, :, None],
                dim=1,
            )
            points = torch.cat((self.positions, samples), dim=0)
            minimum = torch.full(
                (len(points),),
                float("inf"),
                dtype=self.dtype,
                device=self.device,
            )
            for collider_index in range(len(self._collider_centers)):
                count = int(self._plane_counts[collider_index])
                if count <= 0:
                    continue
                rotation = self._collider_rotations[collider_index]
                local = (
                    points - self._collider_centers[collider_index]
                ) @ rotation
                signed = torch.max(
                    local @ self._plane_normals[collider_index, :count].T
                    - self._plane_offsets[collider_index, :count][None, :],
                    dim=1,
                ).values - self.config.particle_radius_m
                minimum = torch.minimum(minimum, signed)
            influence = torch.clamp(
                (band_m - minimum) / band_m, 0.0, 1.0
            )
            node_influence = influence[: len(self.positions)].clone()
            sample_influence = influence[len(self.positions) :]
            node_influence.scatter_reduce_(
                0,
                faces.reshape(-1),
                sample_influence[:, None].expand(-1, 3).reshape(-1),
                reduce="amax",
                include_self=True,
            )
            return node_influence

    def maximum_surface_penetration_by_arm(self) -> np.ndarray:
        """Measure current dense proxy penetration for compliant jaw control."""

        import torch

        with torch.no_grad():
            if not len(self._collider_centers):
                return np.zeros(2, dtype=np.float32)
            faces = self._collision_sample_faces
            samples = torch.sum(
                self.positions[faces]
                * self._collision_sample_weights[:, :, None],
                dim=1,
            )
            points = torch.cat(
                (self.positions[self._boundary_indices], samples), dim=0
            )
            penetration_by_collider = []
            for collider_index in range(len(self._collider_centers)):
                count = int(self._plane_counts[collider_index])
                if count <= 0:
                    penetration_by_collider.append(0.0)
                    continue
                rotation = self._collider_rotations[collider_index]
                local = (
                    points - self._collider_centers[collider_index]
                ) @ rotation
                signed = torch.max(
                    local @ self._plane_normals[collider_index, :count].T
                    - self._plane_offsets[collider_index, :count][None, :],
                    dim=1,
                ).values - self.config.particle_radius_m
                penetration_by_collider.append(
                    max(0.0, -float(torch.min(signed)))
                )
            output = np.zeros(len(self._grasp_active_by_arm))
            penetration_array = np.asarray(
                penetration_by_collider, dtype=np.float64
            )
            for arm_index in range(len(output)):
                owned = self._collider_arm_indices == arm_index
                output[arm_index] = max(
                    penetration_array[owned], default=0.0
                )
            return output.astype(np.float32)

    def contact_preload_by_arm(self) -> np.ndarray:
        """Return the weaker persistent finger preload for each gripper."""

        import torch

        with torch.no_grad():
            collider_count = len(self._collider_centers)
            if collider_count == 0:
                return np.zeros(2, dtype=np.float32)
            per_collider = torch.zeros(
                collider_count, dtype=self.dtype, device=self.device
            )
            for values in self._contact_preload_by_set.values():
                if values.numel():
                    per_collider = torch.maximum(
                        per_collider, torch.max(values, dim=0).values
                    )
            output = []
            for arm_index in range(len(self._grasp_active_by_arm)):
                owned = self._collider_arm_indices == arm_index
                fixed_indices = np.flatnonzero(
                    owned & ~self._collider_moving_mask
                )
                moving_indices = np.flatnonzero(
                    owned & self._collider_moving_mask
                )
                fixed = torch.max(
                    per_collider[
                        torch.as_tensor(
                            fixed_indices, dtype=torch.long, device=self.device
                        )
                    ]
                )
                moving = torch.max(
                    per_collider[
                        torch.as_tensor(
                            moving_indices, dtype=torch.long, device=self.device
                        )
                    ]
                )
                output.append(torch.minimum(fixed, moving))
            return (
                torch.stack(output)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    def project_visual_surface_nonpenetration(
        self,
        points: object,
        *,
        clearance_m: float = 0.00025,
        passes: int = 3,
    ):
        """Project the rendered skin outside the authoritative rigid hulls.

        MGPBD intentionally uses a low-dimensional volumetric proxy while the
        TRELLIS appearance has hundreds of thousands of vertices.  Smooth
        Gaussian skinning alone cannot reproduce a sharp finger contact and
        may leave the rendered exterior inside a collider even when proxy
        contact is active.  This final unilateral skin projection supplies
        the missing contact boundary condition without adding a grasp tether
        or changing the MGPBD material state.
        """

        import torch

        if clearance_m < 0.0 or passes < 1:
            raise ValueError("visual contact projection settings are invalid")
        with torch.no_grad():
            result = points.detach().reshape(-1, 3).clone()
            maximum_pre = 0.0
            projected = torch.zeros(
                len(result), dtype=torch.bool, device=result.device
            )
            for pass_index in range(passes):
                table_penetration = self.table_height_m - result[:, 2]
                table_active = table_penetration > 0.0
                if bool(torch.any(table_active)):
                    result[table_active, 2] = self.table_height_m
                    projected |= table_active
                for collider_index in range(len(self._collider_centers)):
                    count = int(self._plane_counts[collider_index])
                    if count <= 0:
                        continue
                    rotation = self._collider_rotations[collider_index]
                    local = (
                        result - self._collider_centers[collider_index]
                    ) @ rotation
                    normals = self._plane_normals[collider_index, :count]
                    offsets = self._plane_offsets[collider_index, :count]
                    signed, plane_index = torch.max(
                        local @ normals.T - offsets[None, :], dim=1
                    )
                    if pass_index == 0 and len(signed):
                        maximum_pre = max(
                            maximum_pre,
                            max(0.0, -float(torch.min(signed))),
                        )
                    active = signed < clearance_m
                    if not bool(torch.any(active)):
                        continue
                    world_normal = normals[plane_index[active]] @ rotation.T
                    result[active] += (
                        clearance_m - signed[active]
                    )[:, None] * world_normal
                    projected |= active

            maximum_post = 0.0
            for collider_index in range(len(self._collider_centers)):
                count = int(self._plane_counts[collider_index])
                if count <= 0:
                    continue
                rotation = self._collider_rotations[collider_index]
                local = (
                    result - self._collider_centers[collider_index]
                ) @ rotation
                signed = torch.max(
                    local @ self._plane_normals[collider_index, :count].T
                    - self._plane_offsets[collider_index, :count][None, :],
                    dim=1,
                ).values
                if len(signed):
                    maximum_post = max(
                        maximum_post,
                        max(0.0, -float(torch.min(signed))),
                    )
            self._visual_contact_projection = {
                "enabled": True,
                "method": "unilateral_projection_against_Genesis_convex_hulls",
                "passes": passes,
                "clearance_m": clearance_m,
                "projected_vertices": int(torch.count_nonzero(projected)),
                "maximum_pre_projection_penetration_m": maximum_pre,
                "maximum_post_projection_penetration_m": maximum_post,
                "physics_state_modified": False,
                "synthetic_attachment": False,
            }
            return result

    def frame_contact_counts(self) -> np.ndarray:
        return self._frame_contact_counts.copy()

    def persistent_grasp_active_by_arm(self) -> np.ndarray:
        """Return contact-persistence state used by rigid-null transport."""

        return self._grasp_active_by_arm.copy()

    def measure_contacts(self) -> None:
        previous = self.positions.detach().clone()
        self._project_collisions(self.positions, previous)

    def diagnostics(self) -> dict[str, object]:
        positions = self.positions.detach().cpu().numpy()
        contact_influence = (
            self.contact_influence_by_node().detach().cpu().numpy()
        )
        tets = positions[self.topology.elements]
        signed_six = np.einsum(
            "ij,ij->i",
            np.cross(tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0]),
            tets[:, 3] - tets[:, 0],
        )
        ratio = signed_six / self._rest_signed_six_volume
        sample_positions = np.sum(
            positions[self.topology.boundary_faces.repeat(4, axis=0)]
            * self._collision_sample_weights.detach().cpu().numpy()[:, :, None],
            axis=1,
        )
        if len(self._collider_centers):
            dense_signed = self.signed_distances(
                sample_positions,
                radius_m=float(self.config.particle_radius_m),
            )
            terminal_minimum_signed = np.min(dense_signed, axis=0)
            terminal_maximum_penetration = float(
                max(0.0, -float(np.min(terminal_minimum_signed)))
            )
        else:
            terminal_minimum_signed = np.empty(0, dtype=np.float32)
            terminal_maximum_penetration = 0.0
        return {
            "kind": "mgpbd-plush",
            "solver": "MGPBD_ARAP_dual_UA_AMG_MGPCG",
            "physics_owner": "custom_MGPBD_not_Genesis_FEM",
            "physics_vertices": int(len(positions)),
            "physics_tetrahedra": int(len(self.topology.elements)),
            "boundary_vertices": int(len(self.topology.boundary_indices)),
            "current_center_m": positions.mean(axis=0).tolist(),
            "current_extents_m": np.ptp(positions, axis=0).tolist(),
            "volume_ratio": {
                "minimum": float(np.min(ratio)),
                "p05": float(np.percentile(ratio, 5)),
                "median": float(np.median(ratio)),
                "p95": float(np.percentile(ratio, 95)),
                "inverted_tetrahedra": int(np.count_nonzero(ratio <= 0.0)),
            },
            "volume_orientation_barrier": {
                "minimum_signed_volume_ratio": (
                    self.config.minimum_signed_volume_ratio
                ),
                "activations": self._volume_barrier_activations,
                "minimum_step_fraction": (
                    self._volume_barrier_minimum_step_fraction
                ),
                "worst_unlimited_candidate_ratio": (
                    self._volume_barrier_worst_candidate_ratio
                ),
            },
            "table_contact": {
                "height_m": self.table_height_m,
                "frame_vertices": self._table_contacts,
                "peak_vertices": self._table_contacts_peak,
            },
            "gripper_contact_count_by_collider": self._frame_contact_counts.tolist(),
            "gripper_contact_peak_by_collider": self._contact_count_peak.tolist(),
            "gripper_contact_groups": {
                "arm_indices": self._collider_arm_indices.tolist(),
                "moving_mask": self._collider_moving_mask.tolist(),
            },
            "dense_surface_contact": {
                "samples_per_boundary_face": 4,
                "sample_count": int(len(self._collision_sample_faces)),
                "frame_sample_contacts_by_collider": (
                    self._frame_sample_contact_counts.tolist()
                ),
                "peak_sample_contacts_by_collider": (
                    self._sample_contact_count_peak.tolist()
                ),
                "maximum_sweep_margin_m": self.config.maximum_sweep_margin_m,
                "frame_maximum_normal_sweep_m": (
                    self._frame_maximum_normal_sweep_m
                ),
                "peak_maximum_normal_sweep_m": (
                    self._peak_maximum_normal_sweep_m
                ),
                "contact_frame": "persistent_local_collider_anchor",
                "two_finger_coarse_transfer": {
                    "gain": self.config.two_finger_coarse_transfer_gain,
                    "closure_threshold": (
                        self.config.two_finger_transfer_closure_threshold
                    ),
                    "transported_frames": self._coarse_grasp_transport_frames,
                    "peak_rigid_motion_m": self._coarse_grasp_transport_peak_m,
                    "cumulative_rigid_motion_m": (
                        self._coarse_grasp_transport_cumulative_m.tolist()
                    ),
                    "center_lock_frames": self._coarse_grasp_center_lock_frames,
                    "peak_center_correction_m": (
                        self._coarse_grasp_center_lock_peak_m
                    ),
                    "cumulative_center_correction_m": (
                        self._coarse_grasp_center_lock_cumulative_m.tolist()
                    ),
                    "active_by_arm": self._grasp_active_by_arm.tolist(),
                    "contact_persistence_frames": (
                        self.config.grasp_contact_persistence_frames
                    ),
                    "miss_frames_by_arm": (
                        self._grasp_miss_frames_by_arm.tolist()
                    ),
                    "driver": "fixed_finger_contact_proxy_centroid_translation",
                    "synthetic_attachment": False,
                },
                "terminal_minimum_signed_distance_m_by_collider": (
                    terminal_minimum_signed.tolist()
                ),
                "terminal_maximum_penetration_m": (
                    terminal_maximum_penetration
                ),
            },
            "visual_surface_contact_projection": dict(
                self._visual_contact_projection
            ),
            "visual_contact_influence": {
                "source": "dense_face_samples_scattered_to_physics_nodes",
                "band_m": 0.012,
                "minimum": float(np.min(contact_influence)),
                "mean": float(np.mean(contact_influence)),
                "maximum": float(np.max(contact_influence)),
                "active_nodes": int(
                    np.count_nonzero(contact_influence > 1.0e-4)
                ),
            },
            "step_ms_last": self._last_step_ms,
            "tetrahedralization": self._tetrahedralization,
            "projection": dict(self.projector.last_metrics),
        }


class MGPBDTetProvider:
    """Expose custom MGPBD vertices through ``FEMTetVisualBinding``'s contract."""

    def __init__(self, solver: MGPBDTetSolver):
        self.solver = solver
        self.elems = solver.topology.elements

    def get_state(self):
        return SimpleNamespace(pos=self.solver.positions)

    def get_particles_pos(self):
        """Expose all volumetric vertices to the smooth visual skinning field."""

        return self.solver.positions


def make_mgpbd_plush_adapter_class():
    """Build the Genesis adapter lazily to keep lightweight imports cheap."""

    from .plush_physics import XPBDPlushObjectAdapter

    class MGPBDPlushObjectAdapter(XPBDPlushObjectAdapter):
        """Reuse verified Genesis convex hull plumbing with MGPBD-owned state."""

        def _include_contact_geom(self, role: str, geom_index: int) -> bool:
            # The gripper link's first collision geom is the wrist servo body,
            # not the fixed fingertip.  Treating it as a finger made an open
            # hand report a valid two-finger grasp before reaching the toy.
            return not ("_fixed_" in role and geom_index == 0)

        def _contact_proxy_vertex_indices(
            self, role: str, geom_index: int, local_vertices: np.ndarray
        ) -> np.ndarray:
            del role, geom_index
            return distal_finger_contact_vertex_indices(
                local_vertices, keep_fraction=0.65
            )

        def configure_grippers(
            self,
            left_fixed: object,
            right_fixed: object,
            left_moving: object,
            right_moving: object,
            left_wrist: object,
            right_wrist: object,
        ) -> None:
            super().configure_grippers(
                left_fixed,
                right_fixed,
                left_moving,
                right_moving,
                left_wrist,
                right_wrist,
            )
            arm_indices = np.asarray(self._collider_arm_indices, dtype=np.int32)
            moving_mask = np.asarray(
                ["_moving_" in role for role in self._collider_roles],
                dtype=bool,
            )
            self.solver.set_collider_groups(arm_indices, moving_mask)

        def record_contact_gate_post_reset(self) -> None:
            if self._contact_gate_injection is None:
                raise RuntimeError("contact gate center was not constructed")
            shell_index = int(
                self._contact_gate_injection["shell_contact_vertex_index"]
            )
            particle = self.solver.shell_positions()[shell_index]
            signed_distances = self.solver.signed_distances(
                particle[None],
                radius_m=float(self.solver.config.particle_radius_m),
            )[0]
            self.solver.measure_contacts()
            self._contact_gate_injection.update(
                post_reset_particle_world_m=particle.tolist(),
                post_reset_particle_signed_distance_m_by_collider=(
                    signed_distances.tolist()
                ),
                post_reset_mgpbd_contact_counts=(
                    self.solver.frame_contact_counts().tolist()
                ),
            )

        def gripper_contact_evidence(self) -> dict[str, object]:
            result = super().gripper_contact_evidence()
            result["persistent_active_by_arm"] = (
                self.solver.persistent_grasp_active_by_arm().tolist()
            )
            result["contact_persistence_frames"] = (
                self.solver.config.grasp_contact_persistence_frames
            )
            return result

        def get_pos(self) -> np.ndarray:
            from .plush_physics import as_numpy

            _positions, center, rotation = self.binding.rigid_transform()
            rest_center = np.asarray(
                self.solver.topology.rest_positions, dtype=np.float64
            ).mean(axis=0)
            origin = as_numpy(center) - rest_center @ as_numpy(rotation)
            return np.asarray(origin, dtype=np.float32)

        def update_visual(self) -> None:
            self.binding.update()

        def gripper_penetration_by_arm(self) -> np.ndarray:
            return self.solver.maximum_surface_penetration_by_arm()

        def gripper_contact_preload_by_arm(self) -> np.ndarray:
            return self.solver.contact_preload_by_arm()

        def diagnostics(self) -> dict[str, object]:
            result = super().diagnostics()
            result["kind"] = "mgpbd-plush"
            result["solver"] = "MGPBD_ARAP_dual_UA_AMG_MGPCG"
            result["physics_owner"] = "custom_MGPBD_not_Genesis_FEM_or_XPBD_shell"
            if "gripper_colliders" in result:
                result["gripper_colliders"]["transport_adapter"] = (
                    "distal_finger_convex_pad_proxies_from_Genesis_geometry"
                )
            return result

    return MGPBDPlushObjectAdapter
