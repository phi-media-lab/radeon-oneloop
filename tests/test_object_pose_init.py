import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gaussian.object_pose_init import (
    PoseInitError,
    build_manual_ring,
    camera_center_from_world_to_camera,
    canonical_orbit_direction,
    deterministic_confident_sample,
    fit_proper_similarity,
    look_at_world_to_camera,
    sha256_file,
    validate_labeled_camera_layout,
)


def manifest(root: Path):
    views = []
    for index, (label, azimuth) in enumerate(
        (("front", 0.0), ("right", 90.0), ("rear", 180.0), ("left", -90.0))
    ):
        image = root / f"{label}.png"
        mask = root / f"{label}_mask.png"
        image.write_bytes(f"image-{index}".encode())
        mask.write_bytes(f"mask-{index}".encode())
        views.append(
            {
                "id": f"anchor_{label}",
                "view_label": label,
                "source_sha256": f"{index + 1:064x}",
                "tier": "A",
                "provenance": "observed",
                "roles": ["pose"],
                "prepared": True,
                "mask_status": "reviewed_pass",
                "nominal_camera_orbit_deg": {"azimuth": azimuth, "elevation": 0.0},
                "image": {"relpath": image.name, "sha256": sha256_file(image)},
                "hard_mask": {"relpath": mask.name, "sha256": sha256_file(mask)},
                "normalization": {"output_width": 1024, "output_height": 1024},
                "mask_qa": {"foreground_bbox_xyxy": [120, 100, 900, 900]},
            }
        )
    return {
        "asset_name": "test_asset",
        "coordinate_convention": {
            "front_axis": "+Y",
            "up_axis": "+Z",
            "viewer_left_axis": "+X",
            "unit": "m",
            "origin": "plush_body_center",
        },
        "metric_anchor": {
            "kind": "product_specification",
            "dimension": "overall_height",
            "value_m": 0.095,
            "uncertainty_m": 0.005,
            "status": "user_confirmed_metric_anchor",
        },
        "views": views,
    }


class CameraGeometryTests(unittest.TestCase):
    def test_documented_front_direction_is_positive_y(self):
        np.testing.assert_allclose(canonical_orbit_direction(0, 0), [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(canonical_orbit_direction(90, 0), [1, 0, 0], atol=1e-12)

    def test_opencv_look_at_is_right_handed_and_recovers_center(self):
        center = np.asarray([0.0, 0.3, 0.0])
        transform = look_at_world_to_camera(center, np.zeros(3))
        self.assertAlmostEqual(float(np.linalg.det(transform[:3, :3])), 1.0)
        np.testing.assert_allclose(camera_center_from_world_to_camera(transform), center, atol=1e-12)
        np.testing.assert_allclose(transform @ [0, 0, 0, 1], [0, 0, 0.3, 1], atol=1e-12)

    def test_proper_similarity_recovers_known_transform(self):
        source = np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        angle = math.radians(25)
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        )
        target = (2.5 * (rotation @ source.T)).T + [1.0, -2.0, 0.5]
        fit = fit_proper_similarity(source, target)
        self.assertAlmostEqual(fit["scale"], 2.5, places=10)
        self.assertAlmostEqual(fit["determinant"], 1.0, places=10)
        self.assertLess(fit["rmse"], 1e-10)

    def test_label_gate_rejects_wrong_view_order(self):
        nominal = np.asarray(
            [canonical_orbit_direction(value, 0) for value in (0, 90, 180, -90)]
        )
        passed = validate_labeled_camera_layout(
            nominal * 2.0,
            nominal,
            camera_up_vectors=np.tile([0.0, 0.0, 1.0], (4, 1)),
            max_angular_error_deg=5,
            max_radius_cv=0.1,
        )
        self.assertTrue(passed["passed"])
        wrong = validate_labeled_camera_layout(
            nominal[[0, 3, 2, 1]] * 2.0,
            nominal,
            camera_up_vectors=np.tile([0.0, 0.0, 1.0], (4, 1)),
            max_angular_error_deg=5,
            max_radius_cv=0.1,
        )
        self.assertFalse(wrong["passed"])

    def test_planar_layout_without_camera_up_cannot_pass_handedness(self):
        nominal = np.asarray(
            [canonical_orbit_direction(value, 0) for value in (0, 90, 180, -90)]
        )
        result = validate_labeled_camera_layout(
            nominal,
            nominal,
            max_angular_error_deg=5,
            max_radius_cv=0.1,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["up_direction_gate"]["available"])

    def test_confident_sampling_is_deterministic_and_not_score_ranked(self):
        valid = np.ones((20, 20), dtype=bool)
        confidence = np.arange(400, dtype=float).reshape(20, 20)
        first = deterministic_confident_sample(valid, confidence, limit=40, seed=7)
        second = deterministic_confident_sample(valid, confidence, limit=40, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 40)
        self.assertLess(int(first.min()), 100)
        self.assertGreater(int(first.max()), 300)


class ManualRingTests(unittest.TestCase):
    def test_builds_metric_four_camera_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cameras, similarity, quality = build_manual_ring(manifest(root), root, radius_m=0.3)
        self.assertEqual(len(cameras["cameras"]), 4)
        self.assertEqual(cameras["camera_model"], "PINHOLE_OPENCV")
        self.assertEqual(similarity["scale"], 1.0)
        self.assertTrue(quality["all_rotations_proper"])
        expected_focal = 800 * 0.3 / 0.095
        self.assertAlmostEqual(cameras["cameras"][0]["focal_xy_px"][0], expected_focal)

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = manifest(root)
            value["views"][0]["hard_mask"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(PoseInitError, "hash mismatch"):
                build_manual_ring(value, root, radius_m=0.3)

    def test_requires_exactly_four_observed_tier_a_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = manifest(root)
            value["views"][0] = copy.deepcopy(value["views"][0])
            value["views"][0]["provenance"] = "generated"
            with self.assertRaisesRegex(PoseInitError, "prepared observed tier-A"):
                build_manual_ring(value, root, radius_m=0.3)


if __name__ == "__main__":
    unittest.main()
