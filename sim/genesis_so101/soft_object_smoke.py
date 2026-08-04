#!/usr/bin/env python3
"""Qualitative AMD/Genesis PBD drop test for the HIL-derived soft object."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import imageio.v3 as iio
import numpy as np
import torch

from .handover_asset import DEFAULT_CONFIG, DEFAULT_SOFT_MESH, load_spec, sha256_file


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_SOFT_MESH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if not 120 <= args.steps <= 1200:
        raise ValueError("steps must be between 120 and 1200")
    args.output.mkdir(parents=True, exist_ok=True)
    spec = load_spec(args.config)
    if not args.mesh.is_file():
        raise FileNotFoundError(args.mesh)

    import genesis as gs

    gs.init(backend=gs.amdgpu, seed=args.seed)
    if gs.backend != gs.amdgpu:
        raise RuntimeError(f"Genesis did not select AMD GPU: {gs.backend}")
    particle_size = spec.pbd["particle_size_m"]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 240.0, substeps=2),
        pbd_options=gs.options.PBDOptions(
            particle_size=particle_size,
            max_stretch_solver_iterations=8,
            max_bending_solver_iterations=4,
            max_volume_solver_iterations=8,
            lower_bound=(-0.35, -0.35, 0.0),
            upper_bound=(0.35, 0.35, 0.65),
        ),
        vis_options=gs.options.VisOptions(ambient_light=(0.65, 0.65, 0.65)),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    soft = scene.add_entity(
        gs.morphs.Mesh(
            file=str(args.mesh.resolve()),
            pos=(0.0, 0.0, 0.24),
            maxvolume=(particle_size**3) / 2.0,
            force_retet=True,
        ),
        material=gs.materials.PBD.Elastic(
            rho=spec.rigid_density_kg_m3,
            static_friction=spec.static_friction,
            kinetic_friction=spec.kinetic_friction,
            stretch_compliance=spec.pbd["stretch_compliance"],
            bending_compliance=spec.pbd["bending_compliance"],
            volume_compliance=spec.pbd["volume_compliance"],
            stretch_relaxation=spec.pbd["stretch_relaxation"],
            bending_relaxation=spec.pbd["bending_relaxation"],
            volume_relaxation=spec.pbd["volume_relaxation"],
        ),
        surface=gs.surfaces.Default(
            color=(0.94, 0.92, 0.86, 1.0), roughness=0.95, vis_mode="visual"
        ),
    )
    camera = scene.add_camera(
        res=(640, 480),
        pos=(0.42, 0.55, 0.32),
        lookat=(0.0, 0.0, 0.10),
        fov=40,
        GUI=False,
    )
    build_started = time.perf_counter()
    scene.build()
    build_s = time.perf_counter() - build_started

    particle_history = []
    capture_steps = {0, args.steps // 2, args.steps - 1}
    captures = {}
    step_times = []
    for step in range(args.steps):
        begin = time.perf_counter()
        scene.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - begin)
        positions = as_numpy(soft.get_particles_pos()).reshape(-1, 3)
        extents = np.ptp(positions, axis=0)
        particle_history.append(
            {
                "center_z_m": float(positions[:, 2].mean()),
                "min_z_m": float(positions[:, 2].min()),
                "extents_m": extents.tolist(),
            }
        )
        if step in capture_steps:
            rgb, _, _, _ = camera.render(rgb=True)
            path = args.output / f"soft_object_step_{step:04d}.png"
            iio.imwrite(path, as_numpy(rgb).astype(np.uint8))
            captures[str(step)] = path.name

    initial_extent = np.asarray(particle_history[0]["extents_m"])
    final_extent = np.asarray(particle_history[-1]["extents_m"])
    z_extents = np.asarray([item["extents_m"][2] for item in particle_history])
    min_z = min(float(item["min_z_m"]) for item in particle_history)
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "radeon_oneloop.genesis_soft_object_smoke.v1",
        "formal": False,
        "backend": str(gs.backend),
        "device": str(gs.device),
        "torch_device": torch.cuda.get_device_name(0),
        "gcn_arch": str(getattr(props, "gcnArchName", "")),
        "solver": "PBD.Elastic",
        "steps": args.steps,
        "dt_s": 1.0 / 240.0,
        "particles": int(as_numpy(soft.get_particles_pos()).reshape(-1, 3).shape[0]),
        "particle_size_m": particle_size,
        "build_s": build_s,
        "step_ms": {
            "mean": 1000.0 * float(np.mean(step_times)),
            "p95": 1000.0 * float(np.percentile(step_times, 95)),
        },
        "shape": {
            "initial_extents_m": initial_extent.tolist(),
            "final_extents_m": final_extent.tolist(),
            "minimum_vertical_extent_m": float(z_extents.min()),
            "maximum_vertical_compression_ratio": float(1.0 - z_extents.min() / initial_extent[2]),
            "final_extent_ratio": (final_extent / initial_extent).tolist(),
            "minimum_particle_z_m": min_z,
        },
        "priors": {
            "nominal_mass_kg": spec.nominal_mass_kg,
            "density_kg_m3": spec.rigid_density_kg_m3,
            "parameters_calibrated": False,
        },
        "config_sha256": spec.config_sha256,
        "mesh_sha256": sha256_file(args.mesh),
        "captures": captures,
        "interpretation": "Qualitative soft-body feasibility only; compression parameters require a measured force-displacement curve.",
    }
    metrics = args.output / "metrics.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hash_paths = [metrics, *(args.output / name for name in captures.values())]
    (args.output / "hashes.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in sorted(hash_paths)) + "\n",
        encoding="utf-8",
    )
    (args.output / "DONE").touch()
    if os.environ.get("ONELOOP_RUN_DIR"):
        (Path(os.environ["ONELOOP_RUN_DIR"]) / "metrics.json").write_text(
            metrics.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
