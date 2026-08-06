"""Matrix-free SOC-ADMM direction solve for orientation-safe MGPBD.

For each tetrahedron, freeze the closest proper rotation ``R`` at the current
state and constrain the affine deformation update with

``||F - R + K d||_F <= c_work``.

This is a convex 9-D second-order cone per tetrahedron.  Since distance to
``SO(3)`` is bounded by distance to any fixed proper rotation, a radius below
one is also an orientation certificate.  In particular,

``C <= 1 - rho_min**(1/3)`` implies ``det(F) >= rho_min``.

The module is deliberately independent from :mod:`mgpbd_tet`: it contains no
Genesis integration and performs no sparse or CPU fallback solve.  All large
operators are implicit Torch gather/scatter operations suitable for Radeon's
CUDA-compatible ROCm API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Callable


@dataclass(frozen=True)
class SOCADMMConfig:
    """Numerical contract for one convex SOC direction solve."""

    beta: float = 1.0e-3
    scale_beta_by_operator_diagonal: bool = False
    beta_minimum: float = 1.0e-4
    beta_maximum: float = 1.0
    kkt_polish_beta_maximum: float | None = None
    adaptive_beta: bool = True
    beta_update_interval: int = 25
    beta_balance_ratio: float = 5.0
    beta_update_factor: float = 2.0
    work_radius: float = 0.989
    true_arap_maximum: float = 1.0
    minimum_signed_volume_ratio: float = 1.0e-6
    maximum_admm_iterations: int = 2_000
    required_consecutive_gate_passes: int = 3
    admm_primal_tolerance: float = 2.0e-4
    admm_dual_relative_tolerance: float = 5.0e-4
    stationarity_relative_tolerance: float = 5.0e-4
    accepted_dtype_stationarity_safety_factor: float = 0.99
    normal_cone_tolerance: float = 2.0e-5
    coupled_material_relative_tolerance: float = 2.0e-6
    pcg_maximum_iterations: int = 500
    pcg_relative_tolerance: float = 5.0e-6
    pcg_absolute_tolerance: float = 1.0e-10
    pcg_residual_replacement_interval: int = 25

    @property
    def proof_radius(self) -> float:
        return min(
            self.true_arap_maximum,
            1.0 - self.minimum_signed_volume_ratio ** (1.0 / 3.0),
        )

    def validate(self) -> None:
        scalars = (
            self.beta,
            self.beta_minimum,
            self.beta_maximum,
            self.beta_balance_ratio,
            self.beta_update_factor,
            self.work_radius,
            self.true_arap_maximum,
            self.minimum_signed_volume_ratio,
            self.admm_primal_tolerance,
            self.admm_dual_relative_tolerance,
            self.stationarity_relative_tolerance,
            self.accepted_dtype_stationarity_safety_factor,
            self.normal_cone_tolerance,
            self.coupled_material_relative_tolerance,
            self.pcg_relative_tolerance,
            self.pcg_absolute_tolerance,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("SOC-ADMM configuration must be finite")
        if (
            self.beta <= 0.0
            or self.beta_minimum <= 0.0
            or self.beta_maximum < self.beta_minimum
            or not self.beta_minimum <= self.beta <= self.beta_maximum
            or self.beta_balance_ratio <= 1.0
            or self.beta_update_factor <= 1.0
        ):
            raise ValueError("SOC-ADMM beta configuration is invalid")
        if self.kkt_polish_beta_maximum is not None and (
            not math.isfinite(self.kkt_polish_beta_maximum)
            or not self.beta_minimum
            <= self.kkt_polish_beta_maximum
            <= self.beta_maximum
        ):
            raise ValueError("SOC-ADMM KKT-polish beta cap is invalid")
        if not 0.0 < self.minimum_signed_volume_ratio < 1.0:
            raise ValueError("SOC-ADMM minimum volume ratio must be in (0, 1)")
        if not 0.0 < self.true_arap_maximum <= 1.0:
            raise ValueError("SOC-ADMM true ARAP maximum must be in (0, 1]")
        if not 0.0 < self.work_radius < self.proof_radius:
            raise ValueError(
                "SOC-ADMM work radius must leave positive determinant-proof margin"
            )
        if self.admm_primal_tolerance > 0.5 * (
            self.proof_radius - self.work_radius
        ):
            raise ValueError(
                "SOC-ADMM primal tolerance must consume at most half the "
                "determinant-proof margin"
            )
        if min(
            self.admm_primal_tolerance,
            self.admm_dual_relative_tolerance,
            self.stationarity_relative_tolerance,
            self.normal_cone_tolerance,
            self.coupled_material_relative_tolerance,
            self.pcg_relative_tolerance,
            self.pcg_absolute_tolerance,
        ) <= 0.0:
            raise ValueError("SOC-ADMM tolerances must be positive")
        if not 0.0 < self.accepted_dtype_stationarity_safety_factor <= 1.0:
            raise ValueError(
                "accepted-dtype stationarity safety factor must be in (0, 1]"
            )
        if (
            self.maximum_admm_iterations < 1
            or self.required_consecutive_gate_passes < 1
            or self.pcg_maximum_iterations < 1
            or self.pcg_residual_replacement_interval < 1
            or self.beta_update_interval < 1
        ):
            raise ValueError("SOC-ADMM iteration limits must be positive")


@dataclass(frozen=True)
class SOCADMMDirectionResult:
    """Accepted direction, consistent MGPBD multiplier, and audit receipt."""

    direction: object
    delta_lambda: object
    metrics: dict[str, object]


class SOCADMMConvergenceError(RuntimeError):
    """Fail-closed numerical failure carrying the partial audit receipt."""

    def __init__(self, message: str, receipt: dict[str, object]):
        super().__init__(message)
        self.receipt = receipt


def _dynamic_kkt_pcg_target(
    *,
    stationarity_score: float,
    primal_score: float,
    dual_score: float,
    normal_residual: float,
    proof_maximum: float,
    kkt_target_l2: float,
    dual_vector_l2: float,
    config: SOCADMMConfig,
    previous_target_l2: float | None = None,
    polish_active: bool = False,
    confirmation_pass_active: bool = False,
) -> float | None:
    """Allocate the current KKT budget without historical over-tightening.

    Stationarity obeys ``g = -r_pcg + s`` with
    ``s = beta K^T(z_old-z_new)``.  Once the conic gates are close and
    ``||s||`` leaves a positive budget, assigning at most 75% of the
    remainder to the next PCG solve gives the conservative certificate
    ``||r_pcg|| + ||s|| <= kkt_target``.  The target is recomputed from the
    current state and may relax; final exact gates remain authoritative.  Once
    polishing has started, a temporarily exhausted budget must not disable the
    tighter linear solve.  Doing so makes consecutive ADMM iterations
    alternate between a polished solve and the loose RHS-relative default,
    which in turn recreates the dual motion that exhausted the budget.  Keep
    the last force-space target until a new positive budget can replace it.
    """

    close_conic_gates = bool(
        stationarity_score > 1.0
        and primal_score <= config.beta_balance_ratio
        and dual_score <= 1.0
        and normal_residual <= config.normal_cone_tolerance
        and proof_maximum <= config.proof_radius
    )
    remaining = kkt_target_l2 - dual_vector_l2
    candidate = None
    if close_conic_gates and remaining > config.pcg_absolute_tolerance:
        candidate = max(config.pcg_absolute_tolerance, 0.75 * remaining)
    # A first exact gate pass still needs the configured consecutive
    # confirmation.  Do not discard the target that produced that pass before
    # the confirmation solve; it may only stay equal or become tighter.
    if confirmation_pass_active and previous_target_l2 is not None:
        return (
            previous_target_l2
            if candidate is None
            else min(previous_target_l2, candidate)
        )
    if polish_active and previous_target_l2 is not None and candidate is None:
        return previous_target_l2
    return candidate


def _require_tensor(name: str, value: object):
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    return value


def _finite_scalar(value: object) -> float:
    return float(value)


def deformation_gradients(
    positions: object, rest_inverse: object, elements: object
):
    """Return ``F = Ds Dm^-1`` for every tetrahedron."""

    import torch

    positions = _require_tensor("positions", positions)
    rest_inverse = _require_tensor("rest_inverse", rest_inverse)
    elements = _require_tensor("elements", elements)
    tets = positions[elements]
    ds = torch.stack(
        (
            tets[:, 1] - tets[:, 0],
            tets[:, 2] - tets[:, 0],
            tets[:, 3] - tets[:, 0],
        ),
        dim=-1,
    )
    return ds @ rest_inverse


def closest_proper_rotations(deformation: object):
    """Return closest rotations in ``SO(3)`` and their Frobenius distances."""

    import torch

    deformation = _require_tensor("deformation", deformation)
    left, _singular, right_t = torch.linalg.svd(deformation)
    orientation = torch.linalg.det(left @ right_t)
    correction = torch.ones(
        (len(deformation), 3),
        dtype=deformation.dtype,
        device=deformation.device,
    )
    correction[:, -1] = torch.where(
        orientation < 0.0,
        -torch.ones_like(orientation),
        torch.ones_like(orientation),
    )
    rotations = (left * correction[:, None, :]) @ right_t
    distances = torch.linalg.vector_norm(
        deformation - rotations, dim=(1, 2)
    )
    return rotations, distances


def apply_deformation_jacobian(
    direction: object, rest_inverse: object, elements: object
):
    """Apply implicit ``K`` and return one ``3 x 3`` update per tet."""

    return deformation_gradients(direction, rest_inverse, elements)


def vertex_incidence_slots(elements: object, vertex_count: int):
    """Build a padded, deterministic gather map for tet-local values.

    Slot values index ``elements.reshape(-1)``; the final sentinel selects a
    zero row.  Unlike repeated ``index_add_`` calls, reducing the gathered
    incidence dimension has a fixed order and avoids ROCm atomic-add jitter in
    PCG true-residual evaluations.
    """

    import torch

    elements = _require_tensor("elements", elements)
    if (
        elements.dtype != torch.long
        or elements.ndim != 2
        or elements.shape[1] != 4
    ):
        raise TypeError("elements must be a Torch long tensor with shape (T, 4)")
    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    flat = elements.reshape(-1)
    if (
        len(flat) < 1
        or int(torch.min(flat)) < 0
        or int(torch.max(flat)) >= vertex_count
    ):
        raise ValueError("elements contain an out-of-range vertex index")
    sorted_vertices, flat_order = torch.sort(flat)
    counts = torch.bincount(flat, minlength=vertex_count)
    maximum_degree = int(torch.max(counts))
    offsets = torch.cumsum(counts, dim=0) - counts
    ranks = torch.arange(
        len(flat), dtype=torch.long, device=elements.device
    ) - torch.repeat_interleave(offsets, counts)
    sentinel = len(flat)
    slots = torch.full(
        (vertex_count, maximum_degree),
        sentinel,
        dtype=torch.long,
        device=elements.device,
    )
    slots[sorted_vertices, ranks] = flat_order
    return slots


def deterministic_vertex_sum(local_values: object, incidence_slots: object):
    """Sum ``(tet, local, ...)`` values by vertex in fixed gather order."""

    import torch

    local_values = _require_tensor("local_values", local_values)
    incidence_slots = _require_tensor("incidence_slots", incidence_slots)
    if local_values.ndim < 2 or local_values.shape[1] != 4:
        raise ValueError("local_values must have shape (T, 4, ...)")
    if incidence_slots.dtype != torch.long or incidence_slots.ndim != 2:
        raise TypeError("incidence_slots must be a 2-D Torch long tensor")
    if incidence_slots.device != local_values.device:
        raise TypeError("incidence_slots must match local_values device")
    flat = local_values.reshape(-1, *local_values.shape[2:])
    zero = torch.zeros(
        (1, *local_values.shape[2:]),
        dtype=local_values.dtype,
        device=local_values.device,
    )
    padded = torch.cat((flat, zero), dim=0)
    if int(torch.max(incidence_slots)) > len(flat):
        raise ValueError("incidence_slots contains an invalid local-value index")
    gathered = padded[incidence_slots]
    # Spell out a balanced binary tree instead of delegating to a backend
    # reduction whose summation order is not part of the ROCm contract.
    while gathered.shape[1] > 1:
        if gathered.shape[1] % 2:
            padding = torch.zeros_like(gathered[:, :1])
            gathered = torch.cat((gathered, padding), dim=1)
        gathered = gathered[:, 0::2] + gathered[:, 1::2]
    return gathered[:, 0]


def apply_deformation_jacobian_transpose(
    matrices: object,
    rest_inverse: object,
    elements: object,
    vertex_count: int,
    *,
    incidence_slots: object | None = None,
):
    """Apply implicit ``K^T`` by scatter or fixed-order incidence gather."""

    import torch

    matrices = _require_tensor("matrices", matrices)
    rest_inverse = _require_tensor("rest_inverse", rest_inverse)
    elements = _require_tensor("elements", elements)
    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    grad_ds = matrices @ rest_inverse.transpose(1, 2)
    local_values = torch.stack(
        (
            -torch.sum(grad_ds, dim=2),
            grad_ds[:, :, 0],
            grad_ds[:, :, 1],
            grad_ds[:, :, 2],
        ),
        dim=1,
    )
    if incidence_slots is not None:
        return deterministic_vertex_sum(local_values, incidence_slots)
    result = torch.zeros(
        (vertex_count, 3), dtype=matrices.dtype, device=matrices.device
    )
    for local_index in range(4):
        result.index_add_(
            0, elements[:, local_index], local_values[:, local_index]
        )
    return result


def project_frobenius_balls(values: object, radius: float):
    """Project batched ``3 x 3`` values onto identical 9-D Euclidean balls."""

    import torch

    values = _require_tensor("values", values)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("ball radius must be finite and positive")
    norms = torch.linalg.vector_norm(values, dim=(1, 2))
    scale = torch.clamp(
        torch.as_tensor(radius, dtype=values.dtype, device=values.device)
        / norms.clamp_min(torch.finfo(values.dtype).tiny),
        max=1.0,
    )
    return values * scale[:, None, None]


def _apply_material_jacobian(
    gradients: object, direction: object, elements: object
):
    return torch_sum(gradients * direction[elements], dim=(1, 2))


def torch_sum(values: object, *, dim: object):
    """Small indirection keeping Torch imports out of module import time."""

    import torch

    return torch.sum(values, dim=dim)


def _apply_material_jacobian_transpose(
    gradients: object,
    coefficients: object,
    elements: object,
    vertex_count: int,
    *,
    incidence_slots: object | None = None,
):
    import torch

    result = torch.zeros(
        (vertex_count, 3), dtype=gradients.dtype, device=gradients.device
    )
    weighted = coefficients[:, None, None] * gradients
    if incidence_slots is not None:
        return deterministic_vertex_sum(weighted, incidence_slots)
    for local_index in range(4):
        result.index_add_(
            0, elements[:, local_index], weighted[:, local_index]
        )
    return result


def _operator_diagonal_components(
    masses: object,
    gradients: object,
    alpha: object,
    rest_inverse: object,
    elements: object,
    *,
    incidence_slots: object | None = None,
):
    import torch

    vertex_count = len(masses)
    hessian_diagonal = masses[:, None].expand(-1, 3).clone()
    material_local = gradients * gradients / alpha[:, None, None]
    if incidence_slots is not None:
        hessian_diagonal += deterministic_vertex_sum(
            material_local, incidence_slots
        )
    else:
        for local_index in range(4):
            hessian_diagonal.index_add_(
                0, elements[:, local_index], material_local[:, local_index]
            )
    local_coefficients = torch.stack(
        (
            -torch.sum(rest_inverse, dim=1),
            rest_inverse[:, 0],
            rest_inverse[:, 1],
            rest_inverse[:, 2],
        ),
        dim=1,
    )
    deformation_diagonal_tets = torch.sum(
        local_coefficients * local_coefficients, dim=2
    )
    deformation_diagonal = torch.zeros_like(hessian_diagonal)
    if incidence_slots is not None:
        scalar_diagonal = deterministic_vertex_sum(
            deformation_diagonal_tets, incidence_slots
        )
        deformation_diagonal += scalar_diagonal[:, None]
    else:
        for local_index in range(4):
            deformation_diagonal.index_add_(
                0,
                elements[:, local_index],
                deformation_diagonal_tets[:, local_index, None].expand(-1, 3),
            )
    return hessian_diagonal, deformation_diagonal


def _matrix_free_pcg(
    operator: Callable[[object], object],
    rhs: object,
    diagonal: object,
    initial: object,
    config: SOCADMMConfig,
    *,
    target_l2_override: float | None = None,
) -> tuple[object, dict[str, object]]:
    """Jacobi-PCG with true-residual stopping and periodic replacement."""

    import torch

    solution = initial.clone()
    rhs_norm = torch.linalg.vector_norm(rhs)
    rhs_norm_value = _finite_scalar(rhs_norm)
    default_target = max(
        config.pcg_absolute_tolerance,
        config.pcg_relative_tolerance * rhs_norm_value,
    )
    if target_l2_override is not None and (
        not math.isfinite(target_l2_override) or target_l2_override <= 0.0
    ):
        raise ValueError("PCG target override must be finite and positive")
    target = max(
        config.pcg_absolute_tolerance,
        min(
            default_target,
            (
                target_l2_override
                if target_l2_override is not None
                else default_target
            ),
        ),
    )
    residual = rhs - operator(solution)
    residual_is_true = True
    true_norm = torch.linalg.vector_norm(residual)
    true_norm_value = _finite_scalar(true_norm)
    history = [true_norm_value]
    if not math.isfinite(true_norm_value):
        return solution, {
            "converged": False,
            "breakdown": "nonfinite_initial_true_residual",
            "iterations": 0,
            "rhs_l2": rhs_norm_value,
            "target_l2": target,
            "default_target_l2": default_target,
            "target_l2_override": target_l2_override,
            "true_residual_l2": true_norm_value,
            "true_relative_residual": float("inf"),
            "true_residual_to_target": float("inf"),
            "residual_replacements": 0,
            "true_residual_history": history,
        }
    if true_norm_value <= target:
        relative = (
            true_norm_value / rhs_norm_value if rhs_norm_value > 0.0 else 0.0
        )
        return solution, {
            "converged": True,
            "breakdown": None,
            "iterations": 0,
            "rhs_l2": rhs_norm_value,
            "target_l2": target,
            "default_target_l2": default_target,
            "target_l2_override": target_l2_override,
            "true_residual_l2": true_norm_value,
            "true_relative_residual": relative,
            "true_residual_to_target": true_norm_value / target,
            "residual_replacements": 0,
            "true_residual_history": history,
        }

    preconditioned = residual / diagonal
    search = preconditioned.clone()
    residual_preconditioned = torch.sum(residual * preconditioned)
    replacements = 0
    breakdown: str | None = None
    iterations = 0
    for iteration in range(1, config.pcg_maximum_iterations + 1):
        operator_search = operator(search)
        denominator = torch.sum(search * operator_search)
        denominator_value = _finite_scalar(denominator)
        if not math.isfinite(denominator_value) or denominator_value <= 0.0:
            breakdown = "nonpositive_or_nonfinite_curvature"
            break
        step = residual_preconditioned / denominator
        solution = solution + step * search
        residual = residual - step * operator_search
        residual_is_true = False
        iterations = iteration
        recursive_norm_value = _finite_scalar(torch.linalg.vector_norm(residual))
        check_true = (
            recursive_norm_value <= target
            or iteration % config.pcg_residual_replacement_interval == 0
            or iteration == config.pcg_maximum_iterations
        )
        if check_true:
            residual = rhs - operator(solution)
            residual_is_true = True
            true_norm_value = _finite_scalar(torch.linalg.vector_norm(residual))
            history.append(true_norm_value)
            if not math.isfinite(true_norm_value):
                breakdown = "nonfinite_true_residual"
                break
            if true_norm_value <= target:
                break
            replacements += 1
            preconditioned = residual / diagonal
            search = preconditioned.clone()
            residual_preconditioned = torch.sum(residual * preconditioned)
            continue
        next_preconditioned = residual / diagonal
        next_residual_preconditioned = torch.sum(
            residual * next_preconditioned
        )
        current_value = _finite_scalar(residual_preconditioned)
        next_value = _finite_scalar(next_residual_preconditioned)
        if (
            not math.isfinite(current_value)
            or not math.isfinite(next_value)
            or current_value <= 0.0
            or next_value <= 0.0
        ):
            breakdown = "nonpositive_or_nonfinite_preconditioned_residual"
            break
        search = next_preconditioned + (
            next_residual_preconditioned / residual_preconditioned
        ) * search
        preconditioned = next_preconditioned
        residual_preconditioned = next_residual_preconditioned

    # When the loop exits on a directly recomputed true residual, retain that
    # exact audited tensor.  Re-evaluating the same ROCm gather/scatter
    # operator here can differ at the last FP32 ulps and turn an already
    # satisfied strict target into a false failure.  Recursive/breakdown exits
    # still receive a fresh true-residual evaluation.
    final_residual = (
        residual if residual_is_true else rhs - operator(solution)
    )
    true_norm_value = _finite_scalar(torch.linalg.vector_norm(final_residual))
    if not history or history[-1] != true_norm_value:
        history.append(true_norm_value)
    relative = (
        true_norm_value / rhs_norm_value
        if rhs_norm_value > 0.0
        else (0.0 if true_norm_value == 0.0 else float("inf"))
    )
    converged = (
        breakdown is None
        and math.isfinite(true_norm_value)
        and true_norm_value <= target
    )
    return solution, {
        "converged": bool(converged),
        "breakdown": breakdown,
        "iterations": int(iterations),
        "rhs_l2": rhs_norm_value,
        "target_l2": target,
        "default_target_l2": default_target,
        "target_l2_override": target_l2_override,
        "true_residual_l2": true_norm_value,
        "true_relative_residual": relative,
        "true_residual_to_target": true_norm_value / target,
        "residual_replacements": int(replacements),
        "true_residual_history": history,
    }


def _validate_inputs(
    current: object,
    rest_inverse: object,
    elements: object,
    masses: object,
    material_gradients: object,
    q: object,
    alpha: object,
) -> tuple[object, object, object, object, object, object, object]:
    import torch

    current = _require_tensor("current", current)
    rest_inverse = _require_tensor("rest_inverse", rest_inverse)
    elements = _require_tensor("elements", elements)
    masses = _require_tensor("masses", masses)
    material_gradients = _require_tensor(
        "material_gradients", material_gradients
    )
    q = _require_tensor("q", q)
    alpha = _require_tensor("alpha", alpha)
    if current.ndim != 2 or current.shape[1] != 3 or len(current) < 4:
        raise ValueError("current must have shape (vertex_count, 3)")
    if current.dtype not in (torch.float32, torch.float64):
        raise TypeError("SOC-ADMM supports float32 or float64 only")
    if elements.dtype != torch.long or elements.ndim != 2 or elements.shape[1] != 4:
        raise TypeError("elements must be a Torch long tensor with shape (T, 4)")
    tet_count = len(elements)
    expected_shapes = {
        "rest_inverse": (tet_count, 3, 3),
        "masses": (len(current),),
        "material_gradients": (tet_count, 4, 3),
        "q": (tet_count,),
        "alpha": (tet_count,),
    }
    values = {
        "rest_inverse": rest_inverse,
        "masses": masses,
        "material_gradients": material_gradients,
        "q": q,
        "alpha": alpha,
    }
    if tet_count < 1:
        raise ValueError("SOC-ADMM requires at least one tetrahedron")
    for name, expected in expected_shapes.items():
        value = values[name]
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if value.device != current.device or value.dtype != current.dtype:
            raise TypeError(f"{name} must match current device and dtype")
    if elements.device != current.device:
        raise TypeError("elements must be on the current tensor device")
    if int(torch.min(elements)) < 0 or int(torch.max(elements)) >= len(current):
        raise ValueError("elements contain an out-of-range vertex index")
    for name, value in (("current", current), *values.items()):
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} must be finite")
    if bool(torch.any(masses <= 0.0)):
        raise ValueError("masses must be positive")
    if bool(torch.any(alpha <= 0.0)):
        raise ValueError("alpha must be positive")
    if bool(torch.any(torch.abs(torch.linalg.det(rest_inverse)) <= 1.0e-20)):
        raise ValueError("rest_inverse contains a singular matrix")
    return (
        current.detach(),
        rest_inverse.detach(),
        elements.detach(),
        masses.detach(),
        material_gradients.detach(),
        q.detach(),
        alpha.detach(),
    )


def _normal_cone_metrics(
    z: object,
    physical_dual: object,
    radius: float,
    *,
    scaled_dual: object | None = None,
):
    """Audit ``y in N_ball(z)`` using the physical multiplier ``y=beta*u``.

    Cone membership is invariant to positive scaling in exact arithmetic, but
    the configured finite residual tolerance is not.  Auditing ADMM's scaled
    dual ``u`` would therefore change the effective gate whenever adaptive
    beta changes.  Use the KKT multiplier ``y`` that also appears in the
    stationarity equation.
    """

    import torch

    z_norm = torch.linalg.vector_norm(z, dim=(1, 2))
    y = physical_dual
    y_norm = torch.linalg.vector_norm(y, dim=(1, 2))
    boundary_tolerance = 32.0 * torch.finfo(z.dtype).eps
    boundary = z_norm >= radius - boundary_tolerance
    normal = z / z_norm.clamp_min(torch.finfo(z.dtype).tiny)[:, None, None]
    radial = torch.sum(y * normal, dim=(1, 2))
    tangential = y - radial[:, None, None] * normal
    tangential_norm = torch.linalg.vector_norm(tangential, dim=(1, 2))
    explicit_violation = torch.where(
        boundary,
        torch.sqrt(tangential_norm * tangential_norm + torch.relu(-radial) ** 2),
        y_norm,
    )
    explicit_l2 = torch.linalg.vector_norm(explicit_violation)
    physical_dual_l2 = torch.linalg.vector_norm(y)
    global_relative = explicit_l2 / physical_dual_l2.clamp_min(1.0)
    maximum_block_relative = torch.max(explicit_violation) / torch.max(
        y_norm
    ).clamp_min(1.0)
    # ``z=P(z+u)`` is the exact scaled-ADMM fixed point.  Preserve it as an
    # algorithm diagnostic, while the KKT gate above follows the physical
    # multiplier y and the independent SciPy oracle's convention.
    projection_argument = y if scaled_dual is None else scaled_dual
    fixed_point = z - project_frobenius_balls(z + projection_argument, radius)
    fixed_point_norm = torch.linalg.vector_norm(fixed_point, dim=(1, 2))
    return {
        "dual_convention": "physical_y_equals_beta_times_scaled_dual",
        "gate_residual": _finite_scalar(
            torch.maximum(global_relative, maximum_block_relative)
        ),
        "global_relative_residual": _finite_scalar(global_relative),
        "maximum_block_relative_residual": _finite_scalar(
            maximum_block_relative
        ),
        "physical_dual_l2": _finite_scalar(physical_dual_l2),
        "boundary_count": int(torch.count_nonzero(boundary)),
        "interior_count": int(len(z) - int(torch.count_nonzero(boundary))),
        "maximum_explicit_violation": _finite_scalar(
            torch.max(explicit_violation)
        ),
        "maximum_projection_fixed_point_residual": _finite_scalar(
            torch.max(fixed_point_norm)
        ),
        "maximum_tangential_dual": _finite_scalar(torch.max(tangential_norm)),
        "minimum_boundary_radial_dual": (
            _finite_scalar(torch.min(radial[boundary]))
            if bool(torch.any(boundary))
            else 0.0
        ),
        "maximum_interior_dual": (
            _finite_scalar(torch.max(y_norm[~boundary]))
            if bool(torch.any(~boundary))
            else 0.0
        ),
    }


def solve_soc_admm_direction(
    *,
    current: object,
    rest_inverse: object,
    elements: object,
    masses: object,
    material_gradients: object,
    q: object,
    alpha: object,
    config: SOCADMMConfig | None = None,
) -> SOCADMMDirectionResult:
    """Solve one fixed-rotation, all-tetrahedron SOC direction problem.

    The returned multiplier is reconstructed from the accepted direction as
    ``delta_lambda = -alpha^-1 (q + J d)``.  The function returns only after
    every numerical, conic, true-ARAP, and determinant gate passes.  Numerical
    failure raises :class:`SOCADMMConvergenceError` with a partial receipt.
    """

    import torch

    cfg = config or SOCADMMConfig()
    cfg.validate()
    (
        current,
        rest_inverse,
        elements,
        masses,
        material_gradients,
        q,
        alpha,
    ) = _validate_inputs(
        current,
        rest_inverse,
        elements,
        masses,
        material_gradients,
        q,
        alpha,
    )
    vertex_count = len(current)
    incidence_slots = vertex_incidence_slots(elements, vertex_count)
    deformation = deformation_gradients(current, rest_inverse, elements)
    fixed_rotations, current_distances = closest_proper_rotations(deformation)
    fixed_offset = deformation - fixed_rotations
    output_dtype = current.dtype
    output_current = current
    output_rest_inverse = rest_inverse
    output_masses = masses
    output_material_gradients = material_gradients
    output_q = q
    output_alpha = alpha
    output_fixed_offset = fixed_offset
    current_maximum = _finite_scalar(torch.max(current_distances))
    base_receipt: dict[str, object] = {
        "schema_version": "radeon_oneloop.mgpbd_soc_admm_direction.v1",
        "backend": "Torch_ROCm_matrix_free_SOC_ADMM",
        "converged": False,
        "fallback_used": False,
        "configuration": {
            **asdict(cfg),
            "proof_radius": cfg.proof_radius,
        },
        "tetrahedra": int(len(elements)),
        "vertices": int(vertex_count),
        "transpose_reduction": "fixed_order_vertex_incidence_gather",
        "maximum_vertex_tet_incidence": int(incidence_slots.shape[1]),
        "fixed_rotation_minimum_determinant": _finite_scalar(
            torch.min(torch.linalg.det(fixed_rotations))
        ),
        "current_true_arap_maximum": current_maximum,
        "current_minimum_signed_volume_ratio": _finite_scalar(
            torch.min(torch.linalg.det(deformation))
        ),
    }
    if current_maximum > cfg.work_radius:
        base_receipt["failure"] = "current_state_outside_work_ball"
        raise SOCADMMConvergenceError(
            "SOC-ADMM current state is outside the 9-D work ball: "
            f"{current_maximum:.9g} > {cfg.work_radius:.9g}",
            base_receipt,
        )

    def apply_k(values: object):
        return apply_deformation_jacobian(values, rest_inverse, elements)

    def apply_kt(values: object):
        return apply_deformation_jacobian_transpose(
            values,
            rest_inverse,
            elements,
            vertex_count,
            incidence_slots=incidence_slots,
        )

    def apply_j(values: object):
        return _apply_material_jacobian(
            material_gradients, values, elements
        )

    def apply_jt(values: object):
        return _apply_material_jacobian_transpose(
            material_gradients,
            values,
            elements,
            vertex_count,
            incidence_slots=incidence_slots,
        )

    hessian_diagonal, deformation_diagonal = (
        _operator_diagonal_components(
            masses,
            material_gradients,
            alpha,
            rest_inverse,
            elements,
            incidence_slots=incidence_slots,
        )
    )
    positive_deformation_diagonal = deformation_diagonal[
        deformation_diagonal > torch.finfo(current.dtype).tiny
    ]
    if len(positive_deformation_diagonal) == 0:
        base_receipt["failure"] = "zero_deformation_operator_diagonal"
        raise SOCADMMConvergenceError(
            "SOC-ADMM deformation operator has no positive diagonal",
            base_receipt,
        )
    hessian_scale = _finite_scalar(torch.median(hessian_diagonal))
    deformation_scale = _finite_scalar(
        torch.median(positive_deformation_diagonal)
    )
    operator_scale_ratio = hessian_scale / deformation_scale
    requested_beta = float(cfg.beta)
    if cfg.scale_beta_by_operator_diagonal:
        beta = min(
            max(requested_beta * operator_scale_ratio, cfg.beta_minimum),
            cfg.beta_maximum,
        )
    else:
        beta = requested_beta
    beta_scaling = {
        "enabled": bool(cfg.scale_beta_by_operator_diagonal),
        "requested_beta": requested_beta,
        "hessian_diagonal_median": hessian_scale,
        "positive_deformation_diagonal_median": deformation_scale,
        "operator_scale_ratio": operator_scale_ratio,
        "initial_effective_beta": beta,
    }
    base_receipt["beta_scaling"] = beta_scaling

    def operator(values: object):
        material_direction = apply_j(values)
        return (
            masses[:, None] * values
            + apply_jt(material_direction / alpha)
            + beta * apply_kt(apply_k(values))
        )

    diagonal = hessian_diagonal + beta * deformation_diagonal
    if not bool(torch.all(torch.isfinite(diagonal))) or bool(
        torch.any(diagonal <= 0.0)
    ):
        base_receipt["failure"] = "invalid_operator_diagonal"
        raise SOCADMMConvergenceError(
            "SOC-ADMM operator diagonal is nonfinite or nonpositive",
            base_receipt,
        )

    objective_rhs = -apply_jt(q / alpha)
    objective_rhs_l2 = _finite_scalar(torch.linalg.vector_norm(objective_rhs))
    objective_rhs_scale = max(objective_rhs_l2, 1.0)
    base_receipt["objective_rhs_l2"] = objective_rhs_l2
    base_receipt["objective_rhs_scale"] = objective_rhs_scale
    direction = torch.zeros_like(current)
    z = fixed_offset.clone()
    scaled_dual = torch.zeros_like(z)
    pcg_receipts: list[dict[str, object]] = []
    primal_history: list[float] = []
    dual_history: list[float] = []
    stationarity_history: list[float] = []
    proof_history: list[float] = []
    normal_cone_history: list[float] = []
    beta_history: list[float] = []
    beta_updates: list[dict[str, object]] = []
    stationarity_denominator_history: list[float] = []
    stationarity_l2_history: list[float] = []
    stationarity_gate_tolerance_history: list[float] = []
    dual_vector_l2_history: list[float] = []
    kkt_residual_upper_bound_history: list[float] = []
    kkt_target_l2_history: list[float] = []
    pcg_target_override_history: list[float | None] = []
    pcg_target_updates: list[dict[str, object]] = []
    pcg_target_override: float | None = None
    kkt_pcg_polish_started_iteration: int | None = None
    kkt_polish_pre_cap_beta: float | None = None
    precision_continuation_active = False
    precision_continuation_started_iteration: int | None = None
    precision_continuation_events: list[dict[str, object]] = []
    converged = False
    consecutive_gate_passes = 0
    consecutive_gate_pass_history: list[int] = []

    for admm_iteration in range(1, cfg.maximum_admm_iterations + 1):
        beta_history.append(beta)
        rhs = objective_rhs + beta * apply_kt(
            z - fixed_offset - scaled_dual
        )
        direction, pcg_receipt = _matrix_free_pcg(
            operator,
            rhs,
            diagonal,
            direction,
            cfg,
            target_l2_override=pcg_target_override,
        )
        pcg_receipt = {**pcg_receipt, "solve_dtype": str(current.dtype)}
        pcg_receipt = {
            "admm_iteration": admm_iteration,
            "effective_beta": beta,
            **pcg_receipt,
        }
        if not pcg_receipt["converged"]:
            can_continue_in_float64 = bool(
                pcg_target_override is not None
                and current.dtype == torch.float32
                and pcg_receipt.get("breakdown") is None
                and math.isfinite(float(pcg_receipt["true_residual_l2"]))
            )
            if can_continue_in_float64 and not precision_continuation_active:
                trigger_receipt = dict(pcg_receipt)
                precision_continuation_active = True
                precision_continuation_started_iteration = admm_iteration
                if (
                    kkt_polish_pre_cap_beta is not None
                    and beta < kkt_polish_pre_cap_beta
                ):
                    old_beta = beta
                    restored_beta = kkt_polish_pre_cap_beta
                    scaled_dual = scaled_dual * (
                        old_beta / restored_beta
                    )
                    beta = restored_beta
                    diagonal = hessian_diagonal + beta * deformation_diagonal
                    beta_history[-1] = beta
                    rhs = objective_rhs + beta * apply_kt(
                        z - fixed_offset - scaled_dual
                    )
                    beta_updates.append(
                        {
                            "after_admm_iteration": int(admm_iteration - 1),
                            "effective_from_admm_iteration": int(
                                admm_iteration
                            ),
                            "old_effective_beta": old_beta,
                            "new_effective_beta": beta,
                            "reason": (
                                "float64_precision_continuation_restore_beta"
                            ),
                            "primal_gate_score": (
                                primal_history[-1]
                                / cfg.admm_primal_tolerance
                                if primal_history
                                else None
                            ),
                            "dual_gate_score": (
                                dual_history[-1]
                                / cfg.admm_dual_relative_tolerance
                                if dual_history
                                else None
                            ),
                            "stationarity_gate_score": (
                                stationarity_history[-1]
                                / cfg.stationarity_relative_tolerance
                                if stationarity_history
                                else None
                            ),
                            "beta_balance_opposition_score": None,
                        }
                    )

                # Promote the complete ADMM state, not only the PCG work
                # vector.  Casting d back to FP32 on every iteration left a
                # reproducible ~4e-4 primal floor even when the FP64 linear
                # solve converged.  The continuation stays in FP64 until all
                # gates pass; the final direction is then cast once and fully
                # re-audited in the caller's dtype.
                current = current.to(dtype=torch.float64)
                rest_inverse = rest_inverse.to(dtype=torch.float64)
                masses = masses.to(dtype=torch.float64)
                material_gradients = material_gradients.to(dtype=torch.float64)
                q = q.to(dtype=torch.float64)
                alpha = alpha.to(dtype=torch.float64)
                fixed_rotations = fixed_rotations.to(dtype=torch.float64)
                fixed_offset = fixed_offset.to(dtype=torch.float64)
                direction = direction.to(dtype=torch.float64)
                z = z.to(dtype=torch.float64)
                scaled_dual = scaled_dual.to(dtype=torch.float64)
                hessian_diagonal, deformation_diagonal = (
                    _operator_diagonal_components(
                        masses,
                        material_gradients,
                        alpha,
                        rest_inverse,
                        elements,
                        incidence_slots=incidence_slots,
                    )
                )
                objective_rhs = -apply_jt(q / alpha)
                objective_rhs_l2 = _finite_scalar(
                    torch.linalg.vector_norm(objective_rhs)
                )
                objective_rhs_scale = max(objective_rhs_l2, 1.0)
                diagonal = hessian_diagonal + beta * deformation_diagonal
                rhs = objective_rhs + beta * apply_kt(
                    z - fixed_offset - scaled_dual
                )
                direction, receipt_64 = _matrix_free_pcg(
                    operator,
                    rhs,
                    diagonal,
                    direction,
                    cfg,
                    target_l2_override=pcg_target_override,
                )
                pcg_receipt = {
                    "admm_iteration": admm_iteration,
                    "effective_beta": beta,
                    **receipt_64,
                    "solve_dtype": "float64",
                    "eventual_output_dtype": str(output_dtype),
                    "precision_continuation_trigger": trigger_receipt,
                }
                precision_continuation_events.append(
                    {
                        "admm_iteration": int(admm_iteration),
                        "reason": "float32_true_residual_floor",
                        "float32_true_residual_l2": float(
                            trigger_receipt["true_residual_l2"]
                        ),
                        "target_l2": float(trigger_receipt["target_l2"]),
                        "float64_true_residual_l2": float(
                            receipt_64["true_residual_l2"]
                        ),
                        "continuation_state_dtype": str(current.dtype),
                        "eventual_output_dtype": str(output_dtype),
                    }
                )
        pcg_receipts.append(pcg_receipt)
        pcg_target_override_history.append(pcg_target_override)
        if not pcg_receipt["converged"]:
            failure = {
                **base_receipt,
                "failure": "pcg_d_step_failed",
                "admm_iterations": admm_iteration,
                "pcg_receipts": pcg_receipts,
                "effective_beta_history": beta_history,
                "adaptive_beta_updates": beta_updates,
                "final_effective_beta": beta,
                "stationarity_relative_history": stationarity_history,
                "stationarity_l2_history": stationarity_l2_history,
                "stationarity_denominator_history": (
                    stationarity_denominator_history
                ),
                "dual_vector_l2_history": dual_vector_l2_history,
                "kkt_residual_upper_bound_history": (
                    kkt_residual_upper_bound_history
                ),
                "kkt_target_l2_history": kkt_target_l2_history,
                "pcg_target_override_history": (
                    pcg_target_override_history
                ),
                "kkt_pcg_polish_started_iteration": (
                    kkt_pcg_polish_started_iteration
                ),
                "final_pcg_target_override": pcg_target_override,
                "pcg_target_updates": pcg_target_updates,
                "precision_continuation_started_iteration": (
                    precision_continuation_started_iteration
                ),
                "precision_continuation_events": precision_continuation_events,
                "kkt_polish_pre_cap_beta": kkt_polish_pre_cap_beta,
            }
            raise SOCADMMConvergenceError(
                "SOC-ADMM matrix-free PCG d-step failed its true-residual gate",
                failure,
            )

        deformation_update = apply_k(direction)
        z_previous = z
        z = project_frobenius_balls(
            fixed_offset + deformation_update + scaled_dual,
            cfg.work_radius,
        )
        primal = fixed_offset + deformation_update - z
        scaled_dual = scaled_dual + primal
        dual_vector = beta * apply_kt(z_previous - z)
        y = beta * scaled_dual
        mass_term = masses[:, None] * direction
        material_state = q + apply_j(direction)
        material_term = apply_jt(material_state / alpha)
        cone_term = apply_kt(y)
        stationarity = mass_term + material_term + cone_term
        stationarity_denominator = (
            torch.linalg.vector_norm(mass_term)
            + torch.linalg.vector_norm(material_term)
            + torch.linalg.vector_norm(cone_term)
        ).clamp_min(torch.finfo(current.dtype).tiny)
        stationarity_l2 = _finite_scalar(torch.linalg.vector_norm(stationarity))
        stationarity_denominator_value = _finite_scalar(
            stationarity_denominator
        )
        stationarity_relative = (
            stationarity_l2 / stationarity_denominator_value
        )
        stationarity_gate_tolerance = cfg.stationarity_relative_tolerance
        if precision_continuation_active and current.dtype != output_dtype:
            stationarity_gate_tolerance *= (
                cfg.accepted_dtype_stationarity_safety_factor
            )
        dual_vector_l2 = _finite_scalar(torch.linalg.vector_norm(dual_vector))
        dual_relative = _finite_scalar(
            torch.linalg.vector_norm(dual_vector)
        ) / objective_rhs_scale
        primal_per_tet = torch.linalg.vector_norm(primal, dim=(1, 2))
        primal_maximum = _finite_scalar(torch.max(primal_per_tet))
        z_norm = torch.linalg.vector_norm(z, dim=(1, 2))
        proof_maximum = _finite_scalar(
            torch.max(z_norm + primal_per_tet)
        )
        normal = _normal_cone_metrics(
            z,
            y,
            cfg.work_radius,
            scaled_dual=scaled_dual,
        )
        normal_residual = float(normal["gate_residual"])
        primal_history.append(primal_maximum)
        dual_history.append(dual_relative)
        stationarity_history.append(stationarity_relative)
        stationarity_denominator_history.append(
            stationarity_denominator_value
        )
        stationarity_l2_history.append(stationarity_l2)
        dual_vector_l2_history.append(dual_vector_l2)
        # For this ADMM splitting, final stationarity is exactly
        # ``-r_pcg + beta K^T(z_old-z_new)`` up to floating-point roundoff.
        # The triangle bound is therefore a conservative a-posteriori KKT
        # certificate and, crucially, exposes when the default relative-RHS
        # PCG target is too loose for the actual stationarity gate.
        kkt_upper_bound = (
            float(pcg_receipt["true_residual_l2"]) + dual_vector_l2
        )
        kkt_target_l2 = (
            stationarity_gate_tolerance * stationarity_denominator_value
        )
        kkt_residual_upper_bound_history.append(kkt_upper_bound)
        kkt_target_l2_history.append(kkt_target_l2)
        stationarity_gate_tolerance_history.append(
            stationarity_gate_tolerance
        )
        proof_history.append(proof_maximum)
        normal_cone_history.append(normal_residual)
        if os.environ.get("ONELOOP_MGPBD_SOC_PROGRESS") == "1" and (
            admm_iteration == 1 or admm_iteration % 25 == 0
        ):
            print(
                "MGPBD_SOC_ROCM_PROGRESS "
                f"iteration={admm_iteration} beta={beta:.9g} "
                f"pcg_iterations={pcg_receipt['iterations']} "
                f"primal_block={primal_maximum:.9g} "
                f"dual_relative={dual_relative:.9g} "
                f"stationarity_relative={stationarity_relative:.9g} "
                f"proof_radius={proof_maximum:.9g}",
                flush=True,
            )
        gate_passed = bool(
            primal_maximum <= cfg.admm_primal_tolerance
            and dual_relative <= cfg.admm_dual_relative_tolerance
            and stationarity_relative
            <= stationarity_gate_tolerance
            and normal_residual <= cfg.normal_cone_tolerance
            and proof_maximum <= cfg.proof_radius
        )
        consecutive_gate_passes = (
            consecutive_gate_passes + 1 if gate_passed else 0
        )
        consecutive_gate_pass_history.append(consecutive_gate_passes)
        if (
            consecutive_gate_passes
            >= cfg.required_consecutive_gate_passes
        ):
            converged = True
            break

        # A RHS-relative PCG target alone is invalid here: the augmented RHS
        # can be hundreds of times larger than the final stationarity
        # denominator.  Recompute a force-space target from the *current* KKT
        # budget.  It may tighten or relax; the exact stationarity gate below
        # remains authoritative.  Beta stays frozen after polish first starts
        # so the operator and warm state do not move underneath the audit.
        primal_score = primal_maximum / cfg.admm_primal_tolerance
        dual_score = dual_relative / cfg.admm_dual_relative_tolerance
        stationarity_score = (
            stationarity_relative / stationarity_gate_tolerance
        )
        old_pcg_target_override = pcg_target_override
        pcg_target_override = _dynamic_kkt_pcg_target(
            stationarity_score=stationarity_score,
            primal_score=primal_score,
            dual_score=dual_score,
            normal_residual=normal_residual,
            proof_maximum=proof_maximum,
            kkt_target_l2=kkt_target_l2,
            dual_vector_l2=dual_vector_l2,
            config=cfg,
            previous_target_l2=old_pcg_target_override,
            polish_active=kkt_pcg_polish_started_iteration is not None,
            confirmation_pass_active=consecutive_gate_passes > 0,
        )
        if pcg_target_override is not None:
            if kkt_pcg_polish_started_iteration is None:
                kkt_pcg_polish_started_iteration = admm_iteration + 1
                polish_beta_cap = cfg.kkt_polish_beta_maximum
                if polish_beta_cap is not None and beta > polish_beta_cap:
                    old_beta = beta
                    kkt_polish_pre_cap_beta = old_beta
                    # Preserve y=beta*u while moving to the empirically
                    # conditioned linear system used for the precision pass.
                    scaled_dual = scaled_dual * (
                        old_beta / polish_beta_cap
                    )
                    beta = polish_beta_cap
                    diagonal = hessian_diagonal + beta * deformation_diagonal
                    beta_updates.append(
                        {
                            "after_admm_iteration": int(admm_iteration),
                            "old_effective_beta": old_beta,
                            "new_effective_beta": beta,
                            "reason": "kkt_polish_conditioning_cap",
                            "primal_gate_score": primal_score,
                            "dual_gate_score": dual_score,
                            "stationarity_gate_score": stationarity_score,
                            "beta_balance_opposition_score": max(
                                dual_score, stationarity_score
                            ),
                        }
                    )
                    consecutive_gate_passes = 0
        if pcg_target_override != old_pcg_target_override:
            pcg_target_updates.append(
                {
                    "after_admm_iteration": int(admm_iteration),
                    "old_target_l2": old_pcg_target_override,
                    "new_target_l2": pcg_target_override,
                    "reason": "current_kkt_budget_dynamic_forcing",
                    "primal_gate_score": primal_score,
                    "dual_gate_score": dual_score,
                    "stationarity_gate_score": stationarity_score,
                    "kkt_target_l2": kkt_target_l2,
                    "dual_vector_l2": dual_vector_l2,
                }
            )
        if (
            cfg.adaptive_beta
            and kkt_pcg_polish_started_iteration is None
            and admm_iteration % cfg.beta_update_interval == 0
            and admm_iteration < cfg.maximum_admm_iterations
        ):
            # The ADMM dual residual remains the reported dual gate.  The KKT
            # score is a separate conditioning guard: do not raise beta when
            # the d-step is already the dominant unresolved error, because
            # that worsens PCG conditioning without fixing stationarity.
            beta_balance_opposition_score = max(
                dual_score, stationarity_score
            )
            new_beta = beta
            reason: str | None = None
            comparison_floor = torch.finfo(current.dtype).tiny
            if (
                primal_score > 1.0
                and primal_score
                > cfg.beta_balance_ratio
                * max(beta_balance_opposition_score, comparison_floor)
            ):
                new_beta = min(beta * cfg.beta_update_factor, cfg.beta_maximum)
                reason = "primal_gate_dominates"
            elif (
                beta_balance_opposition_score > 1.0
                and beta_balance_opposition_score
                > cfg.beta_balance_ratio
                * max(primal_score, comparison_floor)
            ):
                new_beta = max(beta / cfg.beta_update_factor, cfg.beta_minimum)
                reason = "dual_or_kkt_guard_dominates"
            if new_beta != beta:
                old_beta = beta
                # Preserve the unscaled multiplier y = beta * u.
                scaled_dual = scaled_dual * (old_beta / new_beta)
                beta = new_beta
                diagonal = hessian_diagonal + beta * deformation_diagonal
                beta_updates.append(
                    {
                        "after_admm_iteration": int(admm_iteration),
                        "old_effective_beta": old_beta,
                        "new_effective_beta": beta,
                        "reason": reason,
                        "primal_gate_score": primal_score,
                        "dual_gate_score": dual_score,
                        "stationarity_gate_score": stationarity_score,
                        "beta_balance_opposition_score": (
                            beta_balance_opposition_score
                        ),
                    }
                )
                consecutive_gate_passes = 0
    else:
        admm_iteration = cfg.maximum_admm_iterations

    partial_receipt = {
        **base_receipt,
        "admm_iterations": int(admm_iteration),
        "admm_primal_residual_maximum_history": primal_history,
        "admm_dual_residual_relative_history": dual_history,
        "stationarity_relative_history": stationarity_history,
        "stationarity_l2_history": stationarity_l2_history,
        "stationarity_gate_tolerance_history": (
            stationarity_gate_tolerance_history
        ),
        "stationarity_denominator_history": (
            stationarity_denominator_history
        ),
        "dual_vector_l2_history": dual_vector_l2_history,
        "kkt_residual_upper_bound_history": (
            kkt_residual_upper_bound_history
        ),
        "kkt_target_l2_history": kkt_target_l2_history,
        "pcg_target_override_history": pcg_target_override_history,
        "kkt_pcg_polish_started_iteration": (
            kkt_pcg_polish_started_iteration
        ),
        "final_pcg_target_override": pcg_target_override,
        "pcg_target_updates": pcg_target_updates,
        "precision_continuation_started_iteration": (
            precision_continuation_started_iteration
        ),
        "precision_continuation_events": precision_continuation_events,
        "precision_continuation_active": precision_continuation_active,
        "kkt_polish_pre_cap_beta": kkt_polish_pre_cap_beta,
        "normal_cone_residual_history": normal_cone_history,
        "safety_proof_radius_history": proof_history,
        "consecutive_gate_pass_history": consecutive_gate_pass_history,
        "consecutive_gate_passes_final": consecutive_gate_passes,
        "effective_beta_history": beta_history,
        "adaptive_beta_updates": beta_updates,
        "adaptive_beta_update_count": len(beta_updates),
        "final_effective_beta": beta,
        "pcg_receipts": pcg_receipts,
        "pcg_solves": len(pcg_receipts),
        "pcg_iterations_total": int(
            sum(int(receipt["iterations"]) for receipt in pcg_receipts)
        ),
        "pcg_true_residual_to_target_maximum": max(
            float(receipt["true_residual_to_target"])
            for receipt in pcg_receipts
        ),
        "minimum_operator_diagonal": _finite_scalar(torch.min(diagonal)),
    }
    if not converged:
        partial_receipt["failure"] = "admm_iteration_limit"
        raise SOCADMMConvergenceError(
            "SOC-ADMM did not satisfy primal, dual, stationarity, normal-cone, "
            "and determinant-proof gates within its iteration limit",
            partial_receipt,
        )

    deformation_update = apply_k(direction)
    fixed_candidate = fixed_offset + deformation_update
    primal = fixed_candidate - z
    primal_per_tet = torch.linalg.vector_norm(primal, dim=(1, 2))
    z_norm = torch.linalg.vector_norm(z, dim=(1, 2))
    y = beta * scaled_dual
    mass_term = masses[:, None] * direction
    material_state = q + apply_j(direction)
    material_term = apply_jt(material_state / alpha)
    cone_term = apply_kt(y)
    stationarity = mass_term + material_term + cone_term
    stationarity_denominator = (
        torch.linalg.vector_norm(mass_term)
        + torch.linalg.vector_norm(material_term)
        + torch.linalg.vector_norm(cone_term)
    ).clamp_min(torch.finfo(current.dtype).tiny)
    stationarity_relative = _finite_scalar(
        torch.linalg.vector_norm(stationarity) / stationarity_denominator
    )
    normal = _normal_cone_metrics(
        z,
        y,
        cfg.work_radius,
        scaled_dual=scaled_dual,
    )
    delta_lambda = -material_state / alpha
    coupled_material = material_state + alpha * delta_lambda
    coupled_relative = _finite_scalar(
        torch.linalg.vector_norm(coupled_material)
        / torch.linalg.vector_norm(q).clamp_min(torch.finfo(current.dtype).tiny)
    )
    candidate_positions = current + direction
    candidate_deformation = deformation_gradients(
        candidate_positions, rest_inverse, elements
    )
    _candidate_rotations, true_constraints = closest_proper_rotations(
        candidate_deformation
    )
    signed_ratios = torch.linalg.det(candidate_deformation)
    finite_candidate = bool(torch.all(torch.isfinite(candidate_positions))) and bool(
        torch.all(torch.isfinite(candidate_deformation))
    )
    true_arap_maximum = _finite_scalar(torch.max(true_constraints))
    minimum_signed_ratio = _finite_scalar(torch.min(signed_ratios))
    fixed_candidate_maximum = _finite_scalar(
        torch.max(torch.linalg.vector_norm(fixed_candidate, dim=(1, 2)))
    )
    primal_maximum = _finite_scalar(torch.max(primal_per_tet))
    dual_vector = beta * apply_kt(
        z_previous - z
    )
    dual_relative = _finite_scalar(
        torch.linalg.vector_norm(dual_vector)
    ) / objective_rhs_scale
    proof_maximum = _finite_scalar(torch.max(z_norm + primal_per_tet))
    z_violation = max(_finite_scalar(torch.max(z_norm)) - cfg.work_radius, 0.0)
    initial_objective = 0.5 * _finite_scalar(torch.sum(q * q / alpha))
    final_objective = 0.5 * _finite_scalar(
        torch.sum(masses[:, None] * direction * direction)
        + torch.sum(material_state * material_state / alpha)
    )
    objective_allowance = max(
        1.0e-12,
        cfg.stationarity_relative_tolerance * max(initial_objective, 1.0),
    )
    checks = {
        "admm_converged": True,
        "pcg_true_residuals_satisfied": all(
            bool(receipt["converged"])
            and float(receipt["true_residual_to_target"]) <= 1.0 + 1.0e-6
            for receipt in pcg_receipts
        ),
        "admm_primal_satisfied": (
            primal_maximum <= cfg.admm_primal_tolerance
        ),
        "admm_dual_satisfied": (
            dual_relative <= cfg.admm_dual_relative_tolerance
        ),
        "stationarity_satisfied": (
            stationarity_relative <= cfg.stationarity_relative_tolerance
        ),
        "normal_cone_satisfied": (
            float(normal["gate_residual"]) <= cfg.normal_cone_tolerance
        ),
        "soc_projection_feasible": z_violation <= cfg.admm_primal_tolerance,
        "determinant_proof_satisfied": proof_maximum <= cfg.proof_radius,
        "coupled_material_satisfied": (
            coupled_relative <= cfg.coupled_material_relative_tolerance
        ),
        "true_arap_satisfied": true_arap_maximum <= cfg.true_arap_maximum,
        "true_determinant_satisfied": (
            minimum_signed_ratio >= cfg.minimum_signed_volume_ratio
        ),
        "finite_candidate": finite_candidate,
        "objective_not_above_zero_direction": (
            final_objective <= initial_objective + objective_allowance
        ),
    }
    receipt = {
        **partial_receipt,
        "converged": True,
        "failure": None,
        "admm_primal_residual_maximum": primal_maximum,
        "admm_dual_residual_relative": dual_relative,
        "stationarity_relative": stationarity_relative,
        "normal_cone": normal,
        "soc_z_norm_maximum": _finite_scalar(torch.max(z_norm)),
        "soc_z_violation_maximum": z_violation,
        "fixed_rotation_candidate_norm_maximum": fixed_candidate_maximum,
        "safety_proof_radius_maximum": proof_maximum,
        "safety_proof_margin": cfg.proof_radius - proof_maximum,
        "true_arap_maximum": true_arap_maximum,
        "minimum_signed_volume_ratio": minimum_signed_ratio,
        "inverted_or_collapsed_tetrahedra": int(
            torch.count_nonzero(signed_ratios < cfg.minimum_signed_volume_ratio)
        ),
        "coupled_material_residual_relative": coupled_relative,
        "coupled_material_residual_l2": _finite_scalar(
            torch.linalg.vector_norm(coupled_material)
        ),
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "direction_l2": _finite_scalar(torch.linalg.vector_norm(direction)),
        "delta_lambda_l2": _finite_scalar(
            torch.linalg.vector_norm(delta_lambda)
        ),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    if not receipt["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        receipt["failure"] = "final_fail_closed_gate"
        raise SOCADMMConvergenceError(
            "SOC-ADMM final gate failed: " + ", ".join(failed), receipt
        )

    if direction.dtype != output_dtype:
        output_direction = direction.to(dtype=output_dtype)

        def output_apply_k(values: object):
            return apply_deformation_jacobian(
                values, output_rest_inverse, elements
            )

        def output_apply_kt(values: object):
            return apply_deformation_jacobian_transpose(
                values,
                output_rest_inverse,
                elements,
                vertex_count,
                incidence_slots=incidence_slots,
            )

        def output_apply_j(values: object):
            return _apply_material_jacobian(
                output_material_gradients, values, elements
            )

        def output_apply_jt(values: object):
            return _apply_material_jacobian_transpose(
                output_material_gradients,
                values,
                elements,
                vertex_count,
                incidence_slots=incidence_slots,
            )

        output_z = z.to(dtype=output_dtype)
        output_scaled_dual = scaled_dual.to(dtype=output_dtype)
        output_y = beta * output_scaled_dual
        output_fixed_candidate = output_fixed_offset + output_apply_k(
            output_direction
        )
        output_primal = output_fixed_candidate - output_z
        output_primal_per_tet = torch.linalg.vector_norm(
            output_primal, dim=(1, 2)
        )
        output_z_norm = torch.linalg.vector_norm(output_z, dim=(1, 2))
        output_primal_maximum = _finite_scalar(
            torch.max(output_primal_per_tet)
        )
        output_proof_maximum = _finite_scalar(
            torch.max(output_z_norm + output_primal_per_tet)
        )
        output_z_violation = max(
            _finite_scalar(torch.max(output_z_norm)) - cfg.work_radius,
            0.0,
        )
        output_material_state = output_q + output_apply_j(output_direction)
        output_delta_lambda = -output_material_state / output_alpha
        output_mass_term = output_masses[:, None] * output_direction
        output_material_term = output_apply_jt(
            output_material_state / output_alpha
        )
        output_cone_term = output_apply_kt(output_y)
        output_stationarity = (
            output_mass_term + output_material_term + output_cone_term
        )
        output_stationarity_denominator = (
            torch.linalg.vector_norm(output_mass_term)
            + torch.linalg.vector_norm(output_material_term)
            + torch.linalg.vector_norm(output_cone_term)
        ).clamp_min(torch.finfo(output_dtype).tiny)
        output_stationarity_relative = _finite_scalar(
            torch.linalg.vector_norm(output_stationarity)
            / output_stationarity_denominator
        )
        output_previous_z = z_previous.to(dtype=output_dtype)
        output_dual_vector = beta * output_apply_kt(
            output_previous_z - output_z
        )
        output_objective_rhs = -output_apply_jt(output_q / output_alpha)
        output_objective_rhs_scale = max(
            _finite_scalar(torch.linalg.vector_norm(output_objective_rhs)),
            1.0,
        )
        output_dual_relative = _finite_scalar(
            torch.linalg.vector_norm(output_dual_vector)
        ) / output_objective_rhs_scale
        output_normal = _normal_cone_metrics(
            output_z,
            output_y,
            cfg.work_radius,
            scaled_dual=output_scaled_dual,
        )
        output_coupled_material = (
            output_material_state + output_alpha * output_delta_lambda
        )
        output_coupled_relative = _finite_scalar(
            torch.linalg.vector_norm(output_coupled_material)
            / torch.linalg.vector_norm(output_q).clamp_min(
                torch.finfo(output_dtype).tiny
            )
        )
        output_candidate_positions = output_current + output_direction
        output_candidate_deformation = deformation_gradients(
            output_candidate_positions, output_rest_inverse, elements
        )
        _output_rotations, output_true_constraints = (
            closest_proper_rotations(output_candidate_deformation)
        )
        output_signed_ratios = torch.linalg.det(
            output_candidate_deformation
        )
        output_true_arap_maximum = _finite_scalar(
            torch.max(output_true_constraints)
        )
        output_minimum_signed_ratio = _finite_scalar(
            torch.min(output_signed_ratios)
        )
        output_initial_objective = 0.5 * _finite_scalar(
            torch.sum(output_q * output_q / output_alpha)
        )
        output_final_objective = 0.5 * _finite_scalar(
            torch.sum(
                output_masses[:, None]
                * output_direction
                * output_direction
            )
            + torch.sum(
                output_material_state
                * output_material_state
                / output_alpha
            )
        )
        output_objective_allowance = max(
            1.0e-12,
            cfg.stationarity_relative_tolerance
            * max(output_initial_objective, 1.0),
        )
        output_finite = bool(
            torch.all(torch.isfinite(output_candidate_positions))
        ) and bool(torch.all(torch.isfinite(output_candidate_deformation)))
        output_checks = {
            "admm_primal_satisfied": (
                output_primal_maximum <= cfg.admm_primal_tolerance
            ),
            "admm_dual_satisfied": (
                output_dual_relative <= cfg.admm_dual_relative_tolerance
            ),
            "stationarity_satisfied": (
                output_stationarity_relative
                <= cfg.stationarity_relative_tolerance
            ),
            "normal_cone_satisfied": (
                float(output_normal["gate_residual"])
                <= cfg.normal_cone_tolerance
            ),
            "soc_projection_feasible": (
                output_z_violation <= cfg.admm_primal_tolerance
            ),
            "determinant_proof_satisfied": (
                output_proof_maximum <= cfg.proof_radius
            ),
            "coupled_material_satisfied": (
                output_coupled_relative
                <= cfg.coupled_material_relative_tolerance
            ),
            "true_arap_satisfied": (
                output_true_arap_maximum <= cfg.true_arap_maximum
            ),
            "true_determinant_satisfied": (
                output_minimum_signed_ratio
                >= cfg.minimum_signed_volume_ratio
            ),
            "finite_candidate": output_finite,
            "objective_not_above_zero_direction": (
                output_final_objective
                <= output_initial_objective + output_objective_allowance
            ),
        }
        output_reaudit = {
            "dtype": str(output_dtype),
            "admm_primal_residual_maximum": output_primal_maximum,
            "admm_dual_residual_relative": output_dual_relative,
            "stationarity_relative": output_stationarity_relative,
            "normal_cone": output_normal,
            "soc_z_violation_maximum": output_z_violation,
            "safety_proof_radius_maximum": output_proof_maximum,
            "true_arap_maximum": output_true_arap_maximum,
            "minimum_signed_volume_ratio": output_minimum_signed_ratio,
            "inverted_or_collapsed_tetrahedra": int(
                torch.count_nonzero(
                    output_signed_ratios
                    < cfg.minimum_signed_volume_ratio
                )
            ),
            "coupled_material_residual_relative": output_coupled_relative,
            "initial_objective": output_initial_objective,
            "final_objective": output_final_objective,
            "checks": output_checks,
            "passed": bool(all(output_checks.values())),
        }
        receipt["accepted_dtype_reaudit"] = output_reaudit
        checks["accepted_dtype_reaudit_satisfied"] = bool(
            output_reaudit["passed"]
        )
        receipt["checks"] = checks
        receipt["passed"] = bool(all(checks.values()))
        if not receipt["passed"]:
            receipt["failure"] = "accepted_dtype_reaudit_failed"
            failed = sorted(
                name
                for name, passed in output_checks.items()
                if not passed
            )
            raise SOCADMMConvergenceError(
                "SOC-ADMM accepted-dtype re-audit failed: "
                + ", ".join(failed),
                receipt,
            )
        direction = output_direction
        delta_lambda = output_delta_lambda
    return SOCADMMDirectionResult(
        direction=direction,
        delta_lambda=delta_lambda,
        metrics=receipt,
    )
