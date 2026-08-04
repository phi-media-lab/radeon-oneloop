#!/usr/bin/env python3
"""Calibrate the Genesis contact signal with a millimetre-scale pose sweep.

This program is simulation-only.  It never imports the leader hardware adapter
and never opens a serial port.  The object is held at a sequence of known poses
outside and just inside one face of the left gripper collision AABB.  That makes
the resulting force/effort curve useful for choosing a safe haptic transfer
scale, unlike the legacy deep-penetration smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from radeon_oneloop.contracts import ACTION_NAMES

from .scene import HOME_ACTION, build


DEFAULT_PENETRATION_MM = (-2.0, -1.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def positive_face_sweep_centres(
    link_aabb: Sequence[Sequence[float]],
    object_aabb: Sequence[Sequence[float]],
    penetration_mm: Iterable[float],
    *,
    axis: int = 0,
) -> tuple[tuple[float, float, float], ...]:
    """Return object centres approaching the positive face of ``link_aabb``.

    A negative penetration is an explicit clearance.  At zero, the two AABBs
    are just touching.  The other two coordinates align the AABB centres.
    """

    link = np.asarray(link_aabb, dtype=np.float64)
    obj = np.asarray(object_aabb, dtype=np.float64)
    if link.shape != (2, 3) or obj.shape != (2, 3):
        raise ValueError("link and object AABBs must both have shape (2, 3)")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not np.isfinite(link).all() or not np.isfinite(obj).all():
        raise ValueError("AABBs must be finite")
    if np.any(link[1] <= link[0]) or np.any(obj[1] <= obj[0]):
        raise ValueError("AABB maxima must exceed minima")

    half_extent = (obj[1] - obj[0]) / 2.0
    centre = (link[0] + link[1]) / 2.0
    result = []
    for depth_mm in penetration_mm:
        depth_m = float(depth_mm) / 1000.0
        if not math.isfinite(depth_m):
            raise ValueError("penetration values must be finite")
        sample = centre.copy()
        sample[axis] = link[1, axis] + half_extent[axis] - depth_m
        result.append(tuple(float(value) for value in sample))
    return tuple(result)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _object_contact_count(handles: Any) -> int:
    contacts = handles.left.get_contacts(with_entity=handles.object)
    # Entity-level contact queries expose the collision pairs but Genesis 1.3.1
    # does not include a force field here.  Force is read once from the scene
    # manifold by SO101HandoverTask.haptic_feedback().
    return int(_array(contacts["geom_a"]).reshape(-1).shape[0])


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_sweep(
    samples: Sequence[dict[str, Any]],
    *,
    contact_deadband_n: float,
    max_normalized_effort: float = 0.20,
) -> dict[str, Any]:
    """Summarize a sweep and apply conservative simulation-only gates."""

    if not samples:
        raise ValueError("at least one sweep sample is required")
    baseline = [sample for sample in samples if sample["penetration_mm"] < 0.0]
    positive = [sample for sample in samples if sample["penetration_mm"] > 0.0]
    if not baseline or not positive:
        raise ValueError("sweep requires negative-clearance and positive-penetration samples")

    baseline_peak = max(sample["force_n"]["left_peak"] for sample in baseline)
    contact = [
        sample
        for sample in positive
        if sample["force_n"]["left_median"] >= contact_deadband_n
        and sample["object_contact_count_peak"] > 0
    ]
    finite = all(
        math.isfinite(value)
        for sample in samples
        for value in (
            sample["force_n"]["left_peak"],
            sample["force_n"]["right_peak"],
            sample["joint_effort_abs"]["peak"],
        )
    )
    sane = all(
        sample["force_n"]["left_peak"] < 10_000.0
        and sample["force_n"]["right_peak"] < 10_000.0
        and sample["joint_effort_abs"]["peak"] < 100.0
        for sample in samples
    )
    isolated = max(sample["force_n"]["right_peak"] for sample in samples) < 0.5
    baseline_clear = baseline_peak < contact_deadband_n

    if not 0.0 < max_normalized_effort <= 1.0:
        raise ValueError("max_normalized_effort must be in (0, 1]")
    contact_effort_matrix = np.asarray(
        [
            row
            for sample in contact
            for row in sample["joint_effort_samples"]
        ],
        dtype=np.float64,
    )
    joint_p95 = (
        np.percentile(np.abs(contact_effort_matrix), 95.0, axis=0)
        if contact_effort_matrix.size
        else np.zeros(12, dtype=np.float64)
    )
    # The first physical gate excludes the gripper.  Choose the non-gripper
    # left-arm joint carrying the strongest stable reaction.
    eligible = tuple(range(5))
    ranked = sorted(eligible, key=lambda index: float(joint_p95[index]), reverse=True)
    recommended_index = ranked[0] if ranked and joint_p95[ranked[0]] > 0.0 else None
    recommended_effort_p95 = (
        float(joint_p95[recommended_index])
        if recommended_index is not None
        else None
    )
    # HapticSafetyConfig divides effort by full scale and then caps normalized
    # output.  Therefore p95/full-scale must equal the cap, not 1.0.
    recommended_full_scale = (
        recommended_effort_p95 / max_normalized_effort
        if recommended_effort_p95 is not None
        else None
    )
    accepted = bool(
        finite
        and sane
        and isolated
        and baseline_clear
        and len(contact) >= 2
        and recommended_full_scale is not None
        and recommended_full_scale > 0.0
    )
    return {
        "accepted": accepted,
        "finite": finite,
        "within_protocol_sanity_bounds": sane,
        "negative_clearance_is_quiet": baseline_clear,
        "opposite_arm_isolated": isolated,
        "negative_clearance_peak_force_n": baseline_peak,
        "contact_depths_mm": [sample["penetration_mm"] for sample in contact],
        "contact_depth_count": len(contact),
        "max_normalized_effort": max_normalized_effort,
        "left_joint_effort_p95": {
            ACTION_NAMES[index]: float(joint_p95[index]) for index in eligible
        },
        "left_joint_ranking": [ACTION_NAMES[index] for index in ranked],
        "recommended_first_bench_joint": (
            ACTION_NAMES[recommended_index] if recommended_index is not None else None
        ),
        "recommended_first_bench_motor": (
            ACTION_NAMES[recommended_index]
            .removeprefix("left_")
            .removesuffix(".pos")
            if recommended_index is not None
            else None
        ),
        "recommended_joint_effort_p95": recommended_effort_p95,
        "recommended_simulated_effort_full_scale_p95": recommended_full_scale,
        "recommendation_status": (
            "candidate_for_monitor_only_validation" if accepted else "do_not_apply"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-steps", type=int, default=12)
    parser.add_argument("--contact-deadband-n", type=float, default=0.5)
    parser.add_argument("--max-normalized-effort", type=float, default=0.20)
    parser.add_argument(
        "--penetration-mm",
        type=float,
        nargs="+",
        default=DEFAULT_PENETRATION_MM,
    )
    args = parser.parse_args()
    if args.settle_steps < 4:
        raise ValueError("settle-steps must be at least four")
    if args.contact_deadband_n <= 0.0:
        raise ValueError("contact-deadband-n must be positive")
    depths = tuple(float(value) for value in args.penetration_mm)
    if not any(value < 0.0 for value in depths) or not any(
        value > 0.0 for value in depths
    ):
        raise ValueError("penetration sweep must straddle zero")
    args.output.mkdir(parents=True, exist_ok=True)

    task, handles = build(args.asset_root.resolve(), show_viewer=False)
    try:
        task.reset(HOME_ACTION)
        gripper = handles.left.get_link("gripper")
        link_aabb = _array(gripper.get_AABB()).reshape(2, 3)
        object_aabb = _array(handles.object.get_AABB()).reshape(2, 3)
        centres = positive_face_sweep_centres(
            link_aabb, object_aabb, depths, axis=0
        )
        samples = []
        for depth_mm, centre in zip(depths, centres, strict=True):
            task.reset(HOME_ACTION)
            efforts = []
            forces = []
            contact_counts = []
            for _ in range(args.settle_steps):
                # Re-impose the pose before every physics step: this is a
                # controlled kinematic indentation, not an object drop test.
                handles.object.set_pos(np.asarray(centre, dtype=np.float32))
                handles.object.set_quat(
                    np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
                )
                handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))
                task.step(HOME_ACTION, render=False)
                joint_effort, arm_force = task.haptic_feedback()
                count = _object_contact_count(handles)
                efforts.append(tuple(float(value) for value in joint_effort))
                forces.append(tuple(float(value) for value in arm_force))
                contact_counts.append(count)

            force_array = np.asarray(forces, dtype=np.float64)
            effort_array = np.asarray(efforts, dtype=np.float64)
            # The first quarter contains solver warm-up.  Median values below
            # are taken only from the stable suffix; peaks retain every step.
            stable_start = max(1, args.settle_steps // 4)
            stable_force = force_array[stable_start:]
            samples.append(
                {
                    "penetration_mm": depth_mm,
                    "object_centre_m": list(centre),
                    "object_contact_count_peak": max(contact_counts),
                    "force_n": {
                        "left_median": float(np.median(stable_force[:, 0])),
                        "left_peak": float(np.max(force_array[:, 0])),
                        "right_median": float(np.median(stable_force[:, 1])),
                        "right_peak": float(np.max(force_array[:, 1])),
                    },
                    "joint_effort_abs": {
                        "median": float(np.median(np.abs(effort_array[stable_start:, :6]))),
                        "peak": float(np.max(np.abs(effort_array[:, :6]))),
                    },
                    "joint_effort_samples": [list(row) for row in efforts],
                }
            )

        gate = summarize_sweep(
            samples,
            contact_deadband_n=args.contact_deadband_n,
            max_normalized_effort=args.max_normalized_effort,
        )
        report = {
            "schema_version": "radeon_oneloop.genesis_haptic_contact_calibration.v1",
            "formal": False,
            "backend": str(handles.gs.backend),
            "device": str(handles.gs.device),
            "physical_output_commands": False,
            "serial_devices_opened": False,
            "method": {
                "target": "left_gripper_positive_x_face",
                "penetration_mm": list(depths),
                "settle_steps_per_depth": args.settle_steps,
                "contact_deadband_n": args.contact_deadband_n,
                "max_normalized_effort": args.max_normalized_effort,
                "link_aabb_m": link_aabb.tolist(),
                "object_home_aabb_m": object_aabb.tolist(),
                "joint_effort_names": list(ACTION_NAMES),
            },
            "gate": gate,
            "samples": samples,
            "solver_limit_saturation": task.solver_limit_diagnostics(),
            "note": (
                "Simulation-only calibration. Any candidate scale must pass a separate "
                "monitor-only live gate before physical single-joint output."
            ),
        }
        payload = json.dumps(report, indent=2) + "\n"
        (args.output / "metrics.json").write_text(payload, encoding="utf-8")
        print(payload, end="")
        if not gate["accepted"]:
            raise RuntimeError("haptic contact calibration gate did not pass")
    finally:
        handles.gs.destroy()


if __name__ == "__main__":
    main()
