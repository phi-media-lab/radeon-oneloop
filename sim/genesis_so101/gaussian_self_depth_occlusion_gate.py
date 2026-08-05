#!/usr/bin/env python3
"""Verify proxy-free Gaussian occlusion against Genesis scene depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .gaussian_appearance import (
    SafeAppearanceBinding,
    VkSplatAppearanceRenderer,
    composite_with_gaussian_depth,
    full_geometry_candidate_asset,
    link_segmentation_index,
    entity_segmentation_index,
)
from .handover_asset import DEFAULT_COLLISION_MESH, DEFAULT_MESH
from .scene import OBJECT_START_POS, build


DEPTH_TOLERANCE_M = 0.004


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _segmentation_preview(
    segmentation: np.ndarray, *, gripper_index: int, table_index: int
) -> np.ndarray:
    preview = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    preview[segmentation == table_index] = (40, 80, 220)
    preview[segmentation == gripper_index] = (240, 50, 50)
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so101-asset-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v3 as iio

    task, handles = build(
        args.so101_asset_root.resolve(),
        seed=args.seed,
        show_viewer=False,
        object_visualization=False,
    )
    asset = full_geometry_candidate_asset(args.asset_root)
    asset_audit = asset.validate()
    binding = SafeAppearanceBinding.create(
        lambda: VkSplatAppearanceRenderer(asset, args.vksplat_root)
    )
    table_index = entity_segmentation_index(handles.scene, handles.table)
    left_gripper = handles.left.get_link("gripper")
    gripper_index = link_segmentation_index(
        handles.scene, handles.left, left_gripper
    )

    def capture(name: str, position: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        handles.object.set_pos(position.astype(np.float32))
        handles.object.set_quat(np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32))
        handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))
        rgb, depth, segmentation, _ = handles.front_camera.render(
            rgb=True, depth=True, segmentation=True, force_render=True
        )
        base = _array(rgb).astype(np.uint8)
        scene_depth = _array(depth).astype(np.float32)
        segmentation_array = _array(segmentation)
        result = binding.render_from_genesis(handles.front_camera, handles.object)
        if result.frame is None:
            raise RuntimeError(f"appearance fallback during {name}: {result.error}")
        if result.frame.depth_m is None:
            raise RuntimeError("Gaussian renderer did not provide self depth")
        composite = composite_with_gaussian_depth(
            base,
            scene_depth,
            result.frame,
            depth_tolerance_m=DEPTH_TOLERANCE_M,
        )
        gaussian_depth = result.frame.depth_m
        compositor_support = result.frame.alpha[..., 0] > 1.0e-3
        support = result.frame.alpha[..., 0] >= 0.10
        scene_valid = np.isfinite(scene_depth) & (scene_depth > 0.0)
        gaussian_valid = np.isfinite(gaussian_depth) & (gaussian_depth > 0.0)
        depth_occluded = (
            support
            & scene_valid
            & gaussian_valid
            & (gaussian_depth > scene_depth + DEPTH_TOLERANCE_M)
        )
        visible = composite.effective_alpha[..., 0] >= 0.10
        gripper_occluded = depth_occluded & (segmentation_array == gripper_index)
        table_occluded = depth_occluded & (segmentation_array == table_index)
        difference = np.abs(composite.rgb_u8.astype(np.int16) - base.astype(np.int16))
        preserved_error = (
            int(difference[depth_occluded].max()) if np.any(depth_occluded) else 0
        )
        outside_support_error = (
            int(difference[~compositor_support].max())
            if np.any(~compositor_support)
            else 0
        )
        record = {
            "name": name,
            "object_position_m": position.tolist(),
            "gaussian_support_pixels_alpha_ge_0_10": int(np.count_nonzero(support)),
            "gaussian_depth_resolved_fraction": float(
                np.count_nonzero(support & gaussian_valid)
                / max(np.count_nonzero(support), 1)
            ),
            "visible_gaussian_support_pixels": int(np.count_nonzero(visible)),
            "scene_depth_occluded_support_pixels": int(np.count_nonzero(depth_occluded)),
            "gripper_occluded_support_pixels": int(np.count_nonzero(gripper_occluded)),
            "table_occluded_support_pixels": int(np.count_nonzero(table_occluded)),
            "occluded_pixel_max_rgb_error_u8": preserved_error,
            "outside_gaussian_support_max_rgb_error_u8": outside_support_error,
            "gaussian_alpha_clipped_fraction": composite.gaussian_alpha_clipped_fraction,
            "compositor": composite.compositor,
        }
        arrays = {
            "base": base,
            "composite": composite.rgb_u8,
            "segmentation": _segmentation_preview(
                segmentation_array,
                gripper_index=gripper_index,
                table_index=table_index,
            ),
        }
        return record, arrays

    try:
        camera_position = _array(handles.front_camera.get_pos()).astype(np.float64)
        gripper_position = _array(left_gripper.get_pos()).astype(np.float64)
        camera_ray = gripper_position - camera_position
        camera_ray /= np.linalg.norm(camera_ray)
        candidates = []
        for distance_m in (0.025, 0.04, 0.055, 0.07, 0.085):
            candidates.append(
                capture(
                    f"gripper_candidate_{distance_m:.3f}",
                    gripper_position + camera_ray * distance_m,
                )
            )
        gripper_record, gripper_arrays = max(
            candidates,
            key=lambda item: item[0]["gripper_occluded_support_pixels"],
        )
        gripper_record["name"] = "gripper_occlusion"
        table_candidates = [
            capture(
                f"tabletop_candidate_{height_m:.3f}",
                np.asarray(
                    (OBJECT_START_POS[0], OBJECT_START_POS[1], height_m),
                    dtype=np.float64,
                ),
            )
            for height_m in (0.44, 0.42, 0.40, 0.38)
        ]
        table_record, table_arrays = max(
            table_candidates,
            key=lambda item: item[0]["table_occluded_support_pixels"],
        )
        table_record["name"] = "tabletop_occlusion"
        for case_name, arrays in (
            ("gripper_occlusion", gripper_arrays),
            ("tabletop_occlusion", table_arrays),
        ):
            for image_name, image in arrays.items():
                iio.imwrite(args.output / f"{case_name}_{image_name}.png", image)

        def common(record: dict[str, Any]) -> bool:
            return bool(
                record["gaussian_support_pixels_alpha_ge_0_10"] >= 50
                and record["visible_gaussian_support_pixels"] >= 50
                and record["gaussian_depth_resolved_fraction"] >= 0.95
                and record["occluded_pixel_max_rgb_error_u8"] == 0
                and record["outside_gaussian_support_max_rgb_error_u8"] == 0
                and record["compositor"] == "gaussian_self_depth"
            )
        gripper_passed = common(gripper_record) and (
            gripper_record["gripper_occluded_support_pixels"] >= 5
        )
        tabletop_passed = common(table_record) and (
            table_record["table_occluded_support_pixels"] >= 5
        )
        report = {
            "schema_version": "radeon_oneloop.gaussian_self_depth_occlusion_gate.v1",
            "formal": False,
            "accepted": gripper_passed and tabletop_passed,
            "asset": asset_audit,
            "method": "Gaussian self alpha plus projected-center z-buffer against object-free Genesis scene depth",
            "old_obj_visualization": False,
            "legacy_visual_mesh": {
                "path": str(DEFAULT_MESH.resolve()),
                "loaded": handles.object_mesh_path == DEFAULT_MESH.resolve(),
            },
            "collision_proxy": {
                "path": str(DEFAULT_COLLISION_MESH.resolve()),
                "loaded": handles.object_mesh_path == DEFAULT_COLLISION_MESH.resolve(),
                "visualization": handles.object_visualization,
            },
            "collision_proxy_retained": (
                handles.object_mesh_path == DEFAULT_COLLISION_MESH.resolve()
                and handles.object_visualization is False
            ),
            "depth_tolerance_m": DEPTH_TOLERANCE_M,
            "segmentation_indices": {
                "left_gripper": gripper_index,
                "table": table_index,
            },
            "gripper": {"accepted": gripper_passed, **gripper_record},
            "tabletop": {"accepted": tabletop_passed, **table_record},
            "binding": binding.metrics(),
            "physical_output": False,
        }
        (args.output / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["accepted"]:
            raise RuntimeError("Gaussian self-depth occlusion gate failed")
    finally:
        binding.close()
        try:
            handles.gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
