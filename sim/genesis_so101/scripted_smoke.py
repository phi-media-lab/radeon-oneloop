#!/usr/bin/env python3
"""Run deterministic dual-arm sweeps and headless camera checks on AMD GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from radeon_oneloop.contracts import CAMERA_KEYS, IMAGE_SHAPE_HWC

from .scene import (
    ARM_BASE_SEPARATION_M,
    MODEL_FORWARD_UNIT,
    MODEL_LATERAL_UNIT,
    SHARED_BASE_EULER_DEG,
    build,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--sim-hz", type=float, choices=(60.0, 120.0, 200.0), default=120.0
    )
    parser.add_argument("--workspace-texture", type=Path)
    parser.add_argument("--front-camera-calibration", type=Path)
    parser.add_argument(
        "--video-frames",
        type=int,
        default=0,
        help="Capture this many evenly spaced side-by-side frames (zero disables video).",
    )
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument(
        "--constant-action-json",
        help="Optional 12-value action replay instead of the deterministic sweep.",
    )
    parser.add_argument(
        "--ramp-action-json",
        help="Linearly approach this 12-value action over the run.",
    )
    parser.add_argument(
        "--inject-contact-collider-index",
        type=int,
        help="At the midpoint, place the plush in shallow contact with this collider.",
    )
    parser.add_argument("--contact-overlap-m", type=float, default=0.002)
    parser.add_argument(
        "--object-urdf",
        type=Path,
        help="Rigid object URDF with separate visual and collision geometry.",
    )
    parser.add_argument(
        "--plush-visual-mesh",
        type=Path,
        help=(
            "Textured visual mesh driven by the deformable plush body; "
            "mutually exclusive with --object-urdf."
        ),
    )
    parser.add_argument(
        "--plush-physics-mode",
        choices=("granular", "fem", "xpbd", "mgpbd"),
        default="granular",
    )
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least two")
    if args.video_frames < 0 or args.video_frames > args.steps:
        raise ValueError("video-frames must be between zero and steps")
    if args.video_fps < 1:
        raise ValueError("video-fps must be positive")
    if args.object_urdf is not None and args.plush_visual_mesh is not None:
        raise ValueError("--object-urdf and --plush-visual-mesh are mutually exclusive")
    if args.inject_contact_collider_index is not None and (
        args.plush_visual_mesh is None
        or args.plush_physics_mode not in ("granular", "xpbd", "mgpbd")
    ):
        raise ValueError("contact injection requires a custom plush mode")
    if args.constant_action_json is not None and args.ramp_action_json is not None:
        raise ValueError("constant and ramp action replays are mutually exclusive")
    constant_action = None
    if args.constant_action_json is not None:
        constant_action = np.asarray(
            json.loads(args.constant_action_json), dtype=np.float64
        ).reshape(-1)
        if constant_action.shape != (12,) or not np.isfinite(constant_action).all():
            raise ValueError("constant action must contain 12 finite values")
    ramp_action = None
    if args.ramp_action_json is not None:
        ramp_action = np.asarray(
            json.loads(args.ramp_action_json), dtype=np.float64
        ).reshape(-1)
        if ramp_action.shape != (12,) or not np.isfinite(ramp_action).all():
            raise ValueError("ramp action must contain 12 finite values")
    for name, path in (
        ("workspace texture", args.workspace_texture),
        ("front camera calibration", args.front_camera_calibration),
    ):
        if path is not None and not path.is_file():
            raise ValueError(f"{name} must be an existing file")
    args.output.mkdir(parents=True, exist_ok=True)

    # Preserve the collision-only quarantine as the default.  The only visual
    # opt-in accepted here is a URDF that separates visual and collision meshes.
    object_visualization = False  # object_visualization=False is the safe default
    if args.object_urdf is not None or args.plush_visual_mesh is not None:
        object_visualization = True
    started = time.perf_counter()
    task, handles = build(
        args.asset_root.resolve(),
        seed=args.seed,
        show_viewer=False,
        workspace_texture=args.workspace_texture,
        front_camera_calibration=args.front_camera_calibration,
        object_visualization=object_visualization,
        object_urdf_path=args.object_urdf,
        plush_visual_mesh_path=args.plush_visual_mesh,
        plush_physics_mode=args.plush_physics_mode,
        sim_hz=args.sim_hz,
    )
    build_seconds = time.perf_counter() - started
    initial_inter_arm_contacts = int(
        handles.left.get_contacts(with_entity=handles.right)["geom_a"].shape[0]
    )
    step_times = []
    step_timing_groups: dict[str, list[float]] = {
        "render": [],
        "visual_update_no_render": [],
        "physics_only": [],
    }
    rendered = None
    injected_contact_center = None
    video_frames: list[np.ndarray] = []
    capture_steps = (
        {
            index * (args.steps - 1) // (args.video_frames - 1)
            for index in range(args.video_frames)
        }
        if args.video_frames > 1
        else ({args.steps // 2} if args.video_frames == 1 else set())
    )
    for step in range(args.steps):
        if ramp_action is not None:
            fraction = min((step + 1) / max(args.steps - 1, 1), 1.0)
            start_action = np.asarray(task.default_action(), dtype=np.float64)
            action = (start_action + fraction * (ramp_action - start_action)).tolist()
        elif constant_action is None:
            action = list(task.default_action())
            joint = (step // 140) % 5
            offset = 8.0 * math.sin(2.0 * math.pi * (step % 140) / 140.0)
            action[joint] += offset
            action[6 + joint] += offset
        else:
            action = constant_action.tolist()
        if (
            args.inject_contact_collider_index is not None
            and step == args.steps // 2
        ):
            injected_contact_center = handles.object.contact_gate_center(
                args.inject_contact_collider_index,
                overlap_m=args.contact_overlap_m,
            )
            handles.object.set_pos(injected_contact_center)
            handles.object.set_quat((1.0, 0.0, 0.0, 0.0))
            handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))
            handles.object.record_contact_gate_post_reset()
            handles.object.update_visual()
        begin = time.perf_counter()
        observation = task.step(
            action,
            render=(step in (0, args.steps - 1) or step in capture_steps),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        step_times.append(elapsed)
        if step in (0, args.steps - 1) or step in capture_steps:
            step_timing_groups["render"].append(elapsed)
        elif args.plush_visual_mesh is not None:
            step_timing_groups["visual_update_no_render"].append(elapsed)
        else:
            step_timing_groups["physics_only"].append(elapsed)
        if CAMERA_KEYS[0] in observation:
            rendered = observation
            if step in capture_steps:
                front = np.asarray(observation[CAMERA_KEYS[0]], dtype=np.uint8)
                hand = np.asarray(observation[CAMERA_KEYS[1]], dtype=np.uint8)
                video_frames.append(np.concatenate((front, hand), axis=1))
    if rendered is None:
        rendered = task.observe(render=True)
    for key in CAMERA_KEYS:
        image = np.asarray(rendered[key])
        if image.shape != IMAGE_SHAPE_HWC:
            raise RuntimeError(f"{key} shape mismatch: {image.shape}")
        iio.imwrite(args.output / (key.rsplit(".", 1)[-1] + ".png"), image.astype(np.uint8))
    state = np.asarray(rendered["observation.state"])
    if state.shape != (12,) or not np.isfinite(state).all():
        raise RuntimeError(f"invalid state: shape={state.shape}")
    video_path = None
    if video_frames:
        video_path = args.output / "genesis_dual_camera.mp4"
        iio.imwrite(
            video_path,
            np.stack(video_frames),
            fps=args.video_fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
    props = torch.cuda.get_device_properties(0)
    left_base_visual_bottom_z = float(
        task._array(handles.left.get_link("base").get_vAABB())[0, 2]
    )
    report = {
        "schema_version": "radeon_oneloop.genesis_smoke.v1",
        "formal": False,
        "backend": str(handles.gs.backend),
        "device": str(handles.gs.device),
        "torch_device": torch.cuda.get_device_name(0),
        "gcn_arch": str(getattr(props, "gcnArchName", "")),
        "steps": args.steps,
        "constant_action_replay": (
            constant_action.tolist() if constant_action is not None else None
        ),
        "ramp_action_replay": (
            ramp_action.tolist() if ramp_action is not None else None
        ),
        "contact_injection": {
            "collider_index": args.inject_contact_collider_index,
            "overlap_m": args.contact_overlap_m,
            "injected_center_m": (
                injected_contact_center.tolist()
                if injected_contact_center is not None
                else None
            ),
        },
        "sim_hz": args.sim_hz,
        "scene_layout": {
            "left_base_pos_m": list(handles.left_base_pos),
            "right_base_pos_m": list(handles.right_base_pos),
            "base_separation_m": ARM_BASE_SEPARATION_M,
            "model_forward_unit": list(MODEL_FORWARD_UNIT),
            "model_lateral_unit": list(MODEL_LATERAL_UNIT),
            "arrangement": "side_by_side_parallel",
            "shared_base_euler_deg": list(SHARED_BASE_EULER_DEG),
            "table_top_z_m": handles.table_top_z,
            "table_center_xy_m": list(handles.table_center_xy),
            "table_size_xyz_m": list(handles.table_size_xyz),
            "safety_rail_height_m": handles.safety_rail_height_m,
            "workspace_texture": (
                str(args.workspace_texture.resolve())
                if args.workspace_texture is not None
                else None
            ),
            "front_camera_calibration": (
                str(args.front_camera_calibration.resolve())
                if args.front_camera_calibration is not None
                else None
            ),
            "left_base_visual_bottom_z_m": left_base_visual_bottom_z,
            "left_base_table_gap_m": left_base_visual_bottom_z - handles.table_top_z,
            "initial_inter_arm_contacts": initial_inter_arm_contacts,
            "final_inter_arm_contacts": int(
                handles.left.get_contacts(with_entity=handles.right)["geom_a"].shape[0]
            ),
        },
        "solver_limit_saturation": task.solver_limit_diagnostics(),
        "build_seconds": build_seconds,
        "step_ms": {
            "mean": 1000.0 * float(np.mean(step_times)),
            "p50": 1000.0 * float(np.percentile(step_times, 50)),
            "p95": 1000.0 * float(np.percentile(step_times, 95)),
            "p99": 1000.0 * float(np.percentile(step_times, 99)),
            "groups": {
                name: {
                    "count": len(values),
                    "mean": 1000.0 * float(np.mean(values)) if values else None,
                    "p95": (
                        1000.0 * float(np.percentile(values, 95))
                        if values
                        else None
                    ),
                }
                for name, values in step_timing_groups.items()
            },
        },
        "observation": {
            "state_shape": list(state.shape),
            "camera_shapes": {
                key: list(np.asarray(rendered[key]).shape) for key in CAMERA_KEYS
            },
        },
        "task_success": task.success(),
        "object_asset": {
            "path": str(handles.object_mesh_path),
            "visualization": handles.object_visualization,
            "separate_visual_collision_urdf": args.object_urdf is not None,
            "physics_kind": handles.object_physics_kind,
            "plush_visual_binding": args.plush_visual_mesh is not None,
            "effective_initial_position_m": list(task.object_initial_position()),
            "physics_diagnostics": task.object_physics_diagnostics(),
        },
        "task_success_note": (
            "Joint-sweep smoke validates the scene and contracts; it is not a handover evaluation."
        ),
        "video": {
            "path": str(video_path) if video_path else None,
            "frames": len(video_frames),
            "fps": args.video_fps if video_frames else None,
            "metric_eligible": False,
            "note": "Optional joint-sweep visualization; captured render steps are included in step timings.",
        },
    }
    if handles.object_physics_kind == "granular-plush":
        physics = report["object_asset"]["physics_diagnostics"]
        topology = physics["topology"]
        embedding = physics.get("visual_embedding", {})
        no_render_p95_ms = report["step_ms"]["groups"][
            "visual_update_no_render"
        ]["p95"]
        extents = np.asarray(physics["current_extents_m"], dtype=np.float64)
        checks = {
            "closed_shell_topology": (
                topology["shell_vertices"] >= 100
                and topology["shell_faces"]
                == 2 * topology["shell_vertices"] - 4
            ),
            "explicit_hard_grain_fill": (
                topology["filler_particles"] >= 100
                and physics["fill_model"]
                == "dynamic_all_pair_hard_grain_frictional_jamming"
            ),
            "finite_positive_extent": bool(
                np.isfinite(extents).all() and np.all(extents > 0.0)
            ),
            "tight_rest_shape_support": (
                physics["rest_shape"]["maximum_extent_error_ratio"] <= 0.12
            ),
            "exact_visual_embedding": (
                embedding.get("binding")
                == "custom_xpbd_closed_shell_star_tet_barycentric"
                and embedding.get("visual_vertices", 0) >= 10_000
                and embedding.get("minimum_barycentric_weight", -1.0) >= -3.0e-5
                and embedding.get("reconstruction_error_m_max", 1.0) <= 5.0e-5
            ),
            "real_time_control_budget": (
                no_render_p95_ms is not None and no_render_p95_ms <= 20.0
            ),
            "authoritative_contact_without_tether": (
                physics["authoritative_contact"]
                == "XPBD_external_shell_contact_plus_internal_grains"
                and physics["synthetic_attachment"] is False
            ),
        }
        report["granular_scene_gate"] = {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "maximum_rest_extent_error_ratio": 0.12,
                "maximum_visual_reconstruction_error_m": 5.0e-5,
                "maximum_no_render_p95_step_ms": 20.0,
            },
        }
    if handles.object_physics_kind == "fem-plush":
        physics = report["object_asset"]["physics_diagnostics"]
        embedding = physics["embedding"]
        volume = physics["volume_ratio"]
        surface = physics["visual_surface_quality"]
        no_render_p95_ms = report["step_ms"]["groups"][
            "visual_update_no_render"
        ]["p95"]
        extent_ratio = np.asarray(
            physics["extent_ratio_to_rest"], dtype=np.float64
        )
        checks = {
            "native_genesis_fem": (
                physics["solver"] == "Genesis_FEM_implicit_corotated_FP32"
            ),
            "dense_runtime_tetrahedral_volume": (
                physics["physics_vertices"] >= 900
                and physics["physics_tetrahedra"] >= 3_000
            ),
            "full_trellis_visual": (
                physics["visual_vertices"] >= 200_000
                and embedding["visual_faces"] >= 290_000
            ),
            "exact_visual_embedding": (
                embedding["minimum_barycentric_weight"] >= -3.0e-5
                and embedding["reconstruction_error_m_max"] <= 5.0e-5
            ),
            "zero_inverted_tetrahedra": volume["inverted_tetrahedra"] == 0,
            "zero_visual_face_flips": (
                surface["finite"] is True and surface["flipped_faces"] == 0
            ),
            "tight_rest_shape_support": bool(
                np.max(np.abs(extent_ratio - 1.0)) <= 0.10
            ),
            "sixty_hz_control_budget": (
                no_render_p95_ms is not None and no_render_p95_ms <= 12.5
            ),
            "native_contact_without_attachment": (
                physics["contact"] == "native_legacy_rigid_FEM_surface_contact"
                and physics["synthetic_attachment"] is False
            ),
        }
        report["fem_scene_gate"] = {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "minimum_physics_vertices": 900,
                "minimum_physics_tetrahedra": 3_000,
                "minimum_visual_vertices": 200_000,
                "minimum_visual_faces": 290_000,
                "maximum_visual_reconstruction_error_m": 5.0e-5,
                "maximum_rest_extent_error_ratio": 0.10,
                "maximum_no_render_p95_step_ms": 12.5,
            },
        }
    if args.inject_contact_collider_index is not None:
        physics = report["object_asset"]["physics_diagnostics"]
        target = args.inject_contact_collider_index
        if report["object_asset"]["physics_kind"] == "mgpbd-plush":
            dense = physics["dense_surface_contact"]
            checks = {
                "injected_overlap_detected": (
                    physics["contact_gate_injection"]
                    ["post_reset_particle_signed_distance_m_by_collider"][target]
                    < -0.5 * args.contact_overlap_m
                ),
                "contact_projection_executed": (
                    physics["contact_gate_injection"]
                    ["post_reset_mgpbd_contact_counts"][target]
                    > 0
                ),
                "dense_surface_contact_exercised": (
                    dense["peak_sample_contacts_by_collider"][target] > 0
                ),
                "terminal_dense_surface_nonpenetration": (
                    dense["terminal_maximum_penetration_m"] <= 2.5e-4
                ),
                "zero_inverted_tetrahedra": (
                    physics["volume_ratio"]["inverted_tetrahedra"] == 0
                ),
                "moving_link_world_alignment": (
                    physics["gripper_colliders"]["world_alignment"]
                    ["maximum_vertex_alignment_error_m"]
                    <= 2.0e-5
                ),
            }
        else:
            contacts = physics["contacts"]
            checks = {
                "injected_overlap_detected": (
                    physics["contact_gate_injection"]
                    ["post_reset_particle_signed_distance_m_by_collider"][target]
                    < -0.5 * args.contact_overlap_m
                ),
                "contact_projection_executed": (
                    contacts["contact_projection_events_by_collider"][target] > 0
                ),
                "terminal_particle_nonpenetration": (
                    contacts["frame_max_post_projection_penetration_m"] <= 2.5e-4
                ),
                "terminal_visual_nonpenetration": (
                    physics["visual_contact"]["current_maximum_penetration_m"]
                    <= 2.5e-4
                ),
                "moving_link_world_alignment": (
                    physics["gripper_colliders"]["world_alignment"]
                    ["maximum_vertex_alignment_error_m"]
                    <= 2.0e-5
                ),
            }
        report["contact_injection_gate"] = {
            "passed": all(checks.values()),
            "checks": checks,
        }
    payload = json.dumps(report, indent=2) + "\n"
    (args.output / "metrics.json").write_text(payload, encoding="utf-8")
    if os.environ.get("ONELOOP_RUN_DIR"):
        (Path(os.environ["ONELOOP_RUN_DIR"]) / "metrics.json").write_text(
            payload, encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    if "granular_scene_gate" in report and not report["granular_scene_gate"][
        "passed"
    ]:
        raise SystemExit("granular plush Genesis scene gate failed")
    if "fem_scene_gate" in report and not report["fem_scene_gate"]["passed"]:
        raise SystemExit("native FEM plush Genesis scene gate failed")
    if "contact_injection_gate" in report and not report["contact_injection_gate"][
        "passed"
    ]:
        raise SystemExit("XPBD Genesis contact injection gate failed")


if __name__ == "__main__":
    main()
