#!/usr/bin/env python3
"""Validate Genesis contact-to-joint haptic signals without physical output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scene import HOME_ACTION, build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least two")
    args.output.mkdir(parents=True, exist_ok=True)

    task, handles = build(args.asset_root.resolve(), show_viewer=False)
    task.reset(HOME_ACTION)
    gripper_position = task._array(handles.left.get_link("gripper").get_pos()).reshape(3)
    # Place the HIL-derived rigid proxy into the left gripper collision volume. This
    # is intentionally a simulation-only contact probe; no serial device is
    # opened anywhere in this process.
    handles.object.set_pos(gripper_position.astype(np.float32))
    handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))

    contact_samples = []
    effort_samples = []
    try:
        for _ in range(args.steps):
            task.step(HOME_ACTION, render=False)
            efforts, forces = task.haptic_feedback()
            contact_samples.append(forces)
            effort_samples.append(efforts)
        peak_force = np.max(np.asarray(contact_samples, dtype=np.float64), axis=0)
        peak_effort = float(np.max(np.abs(np.asarray(effort_samples, dtype=np.float64))))
        if peak_force[0] <= 0.0 or peak_effort <= 0.0:
            raise RuntimeError(
                f"contact probe produced no left-arm haptic signal: "
                f"force={peak_force[0]}, effort={peak_effort}"
            )
        report = {
            "schema_version": "radeon_oneloop.genesis_haptic_smoke.v1",
            "formal": False,
            "backend": str(handles.gs.backend),
            "device": str(handles.gs.device),
            "steps": args.steps,
            "peak_contact_force_n": peak_force.tolist(),
            "peak_abs_joint_reaction_effort": peak_effort,
            "solver_limit_saturation": task.solver_limit_diagnostics(),
            "physical_output_commands": False,
            "note": "HIL-object-in-gripper simulation probe; not a calibrated haptic rendering result.",
        }
        payload = json.dumps(report, indent=2) + "\n"
        (args.output / "metrics.json").write_text(payload, encoding="utf-8")
        print(payload, end="")
    finally:
        handles.gs.destroy()


if __name__ == "__main__":
    main()
