import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gaussian.sharp_object_fusion import (
    _generator_ply_paths,
    _surface_donor_fields,
    fit_similarity_trimmed,
    quaternion_from_rotation,
    quaternion_multiply,
    select_best_per_voxel,
)


class SharpObjectFusionTests(unittest.TestCase):
    def test_surface_donor_uses_its_own_max_opacity_layer(self):
        arrays = {
            "x": np.zeros(6),
            "f_dc_0": np.array([10, 11, 12, 20, 21, 22]),
            "f_dc_1": np.array([30, 31, 32, 40, 41, 42]),
            "f_dc_2": np.array([50, 51, 52, 60, 61, 62]),
            "opacity": np.array([0.1, 0.9, 0.2, 0.8, 0.2, 0.1]),
        }
        result = _surface_donor_fields(arrays, layers=3, expected_pixels=2)
        np.testing.assert_array_equal(result["f_dc_0"], [11, 20])
        np.testing.assert_array_equal(result["opacity"], [0.9, 0.8])

    def test_unisharp_manifest_resolves_per_view_plys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = []
            for view_id in ("anchor_front", "anchor_right", "anchor_rear", "anchor_left"):
                relpath = f"inference/neutral_rgb_{view_id}/gaussians.ply"
                path = root / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                outputs.append({"view_id": view_id, "relpath": relpath})
            result = _generator_ply_paths(root, {"model": "UniSHARP", "outputs": outputs})
            self.assertEqual(result["anchor_rear"], root / outputs[2]["relpath"])

    def test_trimmed_similarity_recovers_proper_transform_with_outliers(self):
        generator = np.random.default_rng(7)
        source = generator.normal(size=(1000, 3))
        angle = np.deg2rad(23.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        target = 0.24 * (source @ rotation.T) + np.array([0.01, -0.03, 0.08])
        target[:100] += generator.normal(scale=2.0, size=(100, 3))
        result = fit_similarity_trimmed(source, target)
        self.assertAlmostEqual(result.scale, 0.24, places=6)
        np.testing.assert_allclose(result.rotation, rotation, atol=1.0e-6)
        np.testing.assert_allclose(result.translation, [0.01, -0.03, 0.08], atol=1.0e-6)
        self.assertGreater(result.inliers.sum(), 200)

    def test_rotation_quaternion_left_multiplication(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        quaternion = quaternion_from_rotation(rotation)
        np.testing.assert_allclose(
            quaternion,
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            atol=1.0e-7,
        )
        result = quaternion_multiply(quaternion, np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, quaternion, atol=1.0e-7)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_best_per_voxel_prefers_highest_opacity(self):
        keys = np.array([4, 4, 8, 8, 8])
        opacity = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        chosen = select_best_per_voxel(keys, opacity, np.ones(5, dtype=bool))
        np.testing.assert_array_equal(chosen, [1, 3])


if __name__ == "__main__":
    unittest.main()
