#!/usr/bin/env python3
"""Prove conservative gripper and tabletop occlusion of Gaussian appearance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .gaussian_appearance import (
    SafeAppearanceBinding,
    VkSplatAppearanceRenderer,
    composite_with_proxy_depth,
    entity_segmentation_index,
    link_segmentation_index,
    observed_core_asset,
)
from .scene import OBJECT_START_POS, build


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _segmentation_preview(
    segmentation: np.ndarray,
    *,
    object_index: int,
    gripper_index: int,
    table_index: int,
) -> np.ndarray:
    preview = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    preview[segmentation == table_index] = (40, 80, 220)
    preview[segmentation == gripper_index] = (240, 50, 50)
    preview[segmentation == object_index] = (40, 220, 80)
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so101-asset-root", type=Path, required=True)
    parser.add_argument("--observed-core-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import imageio.v3 as iio

    task, handles = build(
        args.so101_asset_root.resolve(), seed=args.seed, show_viewer=False
    )
    asset = observed_core_asset(args.observed_core_root)
    binding = SafeAppearanceBinding.create(
        lambda: VkSplatAppearanceRenderer(asset, args.vksplat_root)
    )
    object_index = entity_segmentation_index(handles.scene, handles.object)
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
        depth_array = _array(depth).astype(np.float32)
        segmentation_array = _array(segmentation)
        result = binding.render_from_genesis(handles.front_camera, handles.object)
        if result.frame is None:
            raise RuntimeError(f"appearance fallback during {name}: {result.error}")
        object_mask = segmentation_array == object_index
        composite = composite_with_proxy_depth(
            base, depth_array, object_mask, result.frame
        )
        support = result.frame.alpha[..., 0] >= 0.10
        gripper_occluded = support & (segmentation_array == gripper_index)
        table_occluded = support & (segmentation_array == table_index)
        non_object_occluded = support & ~object_mask
        visible_gaussian = support & object_mask
        difference = np.abs(composite.rgb_u8.astype(np.int16) - base.astype(np.int16))
        preserved_error = (
            int(difference[non_object_occluded].max())
            if np.any(non_object_occluded)
            else 0
        )
        record = {
            "name": name,
            "object_position_m": position.tolist(),
            "visible_object_pixels": int(np.count_nonzero(object_mask)),
            "gaussian_support_pixels_alpha_ge_0_10": int(np.count_nonzero(support)),
            "gripper_occluded_support_pixels": int(np.count_nonzero(gripper_occluded)),
            "table_occluded_support_pixels": int(np.count_nonzero(table_occluded)),
            "non_object_occluded_support_pixels": int(np.count_nonzero(non_object_occluded)),
            "visible_gaussian_support_pixels": int(np.count_nonzero(visible_gaussian)),
            "occluded_pixel_max_rgb_error_u8": preserved_error,
            "proxy_depth_valid_fraction": float(
                np.mean(np.isfinite(depth_array[object_mask]) & (depth_array[object_mask] > 0.0))
            ),
            "gaussian_alpha_clipped_fraction": composite.gaussian_alpha_clipped_fraction,
        }
        arrays = {
            "base": base,
            "composite": composite.rgb_u8,
            "segmentation": _segmentation_preview(
                segmentation_array,
                object_index=object_index,
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
            record, arrays = capture(
                f"gripper_candidate_{distance_m:.3f}",
                gripper_position + camera_ray * distance_m,
            )
            candidates.append((record, arrays))
        gripper_record, gripper_arrays = max(
            candidates,
            key=lambda item: item[0]["gripper_occluded_support_pixels"],
        )
        gripper_record["name"] = "gripper_occlusion"

        table_record, table_arrays = capture(
            "tabletop_occlusion", np.asarray(OBJECT_START_POS, dtype=np.float64)
        )
        for case_name, arrays in (
            ("gripper_occlusion", gripper_arrays),
            ("tabletop_occlusion", table_arrays),
        ):
            for image_name, image in arrays.items():
                iio.imwrite(args.output / f"{case_name}_{image_name}.png", image)

        gripper_passed = bool(
            gripper_record["visible_object_pixels"] >= 50
            and gripper_record["visible_gaussian_support_pixels"] >= 50
            and gripper_record["gripper_occluded_support_pixels"] >= 5
            and gripper_record["occluded_pixel_max_rgb_error_u8"] == 0
            and gripper_record["proxy_depth_valid_fraction"] == 1.0
        )
        tabletop_passed = bool(
            table_record["visible_object_pixels"] >= 50
            and table_record["visible_gaussian_support_pixels"] >= 50
            and table_record["table_occluded_support_pixels"] >= 5
            and table_record["occluded_pixel_max_rgb_error_u8"] == 0
            and table_record["proxy_depth_valid_fraction"] == 1.0
        )
        report = {
            "schema_version": "radeon_oneloop.gaussian_occlusion_gate.v1",
            "formal": False,
            "accepted": gripper_passed and tabletop_passed,
            "method": "VkSplat premultiplied RGB/alpha gated by front-most Genesis link segmentation and proxy depth",
            "segmentation_indices": {
                "object": object_index,
                "left_gripper": gripper_index,
                "table": table_index,
            },
            "gripper": {"accepted": gripper_passed, **gripper_record},
            "tabletop": {"accepted": tabletop_passed, **table_record},
            "binding": binding.metrics(),
            "physical_output": False,
            "generated_fill_enabled": False,
        }
        (args.output / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["accepted"]:
            raise RuntimeError("Gaussian occlusion gate failed")
    finally:
        binding.close()
        try:
            handles.gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
