import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.genesis_so101.gaussian_appearance import (
    AppearanceFrame,
    ObservedCoreAsset,
    PinholeCamera,
    SafeAppearanceBinding,
    composite_with_proxy_depth,
    entity_segmentation_index,
    link_segmentation_index,
    object_to_camera_opencv,
    transform_from_pos_quat_wxyz,
    validate_rigid_transform,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GaussianAppearanceTests(unittest.TestCase):
    def test_segmentation_resolvers_support_link_level_scene(self):
        class Value:
            def __init__(self, index):
                self.idx = index

        class Scene:
            segmentation_idx_dict = {3: (7, 11), 4: (7, 12), 5: (8, 13)}

        self.assertEqual(entity_segmentation_index(Scene(), Value(8)), 5)
        self.assertEqual(link_segmentation_index(Scene(), Value(7), Value(12)), 4)

    def test_genesis_opengl_camera_is_converted_to_opencv(self):
        world_camera_gl = np.eye(4)
        world_object = np.eye(4)
        world_object[:3, 3] = (0.25, -0.5, -2.0)
        camera_object_cv = object_to_camera_opencv(world_camera_gl, world_object)
        np.testing.assert_allclose(camera_object_cv[:3, 3], (0.25, 0.5, 2.0))
        np.testing.assert_allclose(
            camera_object_cv[:3, :3], np.diag((1.0, -1.0, -1.0))
        )

    def test_wxyz_quaternion_builds_rigid_transform(self):
        transform = transform_from_pos_quat_wxyz(
            (1.0, 2.0, 3.0),
            (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
        )
        np.testing.assert_allclose(transform[:3, 3], (1.0, 2.0, 3.0))
        np.testing.assert_allclose(
            transform[:3, :3] @ (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            atol=1.0e-7,
        )

    def test_rigid_transform_rejects_scale(self):
        transform = np.eye(4)
        transform[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "scale or shear"):
            validate_rigid_transform(transform)

    def test_asset_validation_binds_all_three_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ply = root / "appearance_observed_canonical.ply"
            cameras = root / "cameras_observed.json"
            provenance = root / "provenance.json"
            ply.write_bytes(b"test-ply")
            cameras.write_text(
                json.dumps({"camera_model": "PINHOLE_OPENCV", "cameras": [{}, {}, {}, {}]}),
                encoding="utf-8",
            )
            provenance.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.observed_core_canonicalization.v1",
                        "output_ply_sha256": _sha(ply),
                        "gaussian_count": 1,
                        "observed_only_training": True,
                        "formal": False,
                        "provenance_class": "observed_core_candidate",
                    }
                ),
                encoding="utf-8",
            )
            asset = ObservedCoreAsset(
                ply,
                cameras,
                provenance,
                expected_ply_sha256=_sha(ply),
                expected_cameras_sha256=_sha(cameras),
                expected_provenance_sha256=_sha(provenance),
                expected_gaussians=1,
                expected_formal=False,
            )
            audit = asset.validate()
            self.assertEqual(audit["gaussian_count"], 1)
            self.assertEqual(audit["camera_count"], 4)
            self.assertFalse(audit["formal"])
            ply.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                asset.validate()

    def test_asset_validation_rejects_unexpected_formal_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ply = root / "appearance_observed_canonical.ply"
            cameras = root / "cameras_observed.json"
            provenance = root / "provenance.json"
            ply.write_bytes(b"test-ply")
            cameras.write_text(
                json.dumps({"camera_model": "PINHOLE_OPENCV", "cameras": []}),
                encoding="utf-8",
            )
            provenance.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.observed_core_canonicalization.v1",
                        "output_ply_sha256": _sha(ply),
                        "gaussian_count": 1,
                        "observed_only_training": True,
                        "formal": False,
                    }
                ),
                encoding="utf-8",
            )
            asset = ObservedCoreAsset(
                ply,
                cameras,
                provenance,
                expected_ply_sha256=_sha(ply),
                expected_cameras_sha256=_sha(cameras),
                expected_provenance_sha256=_sha(provenance),
                expected_gaussians=1,
                expected_formal=True,
            )
            with self.assertRaisesRegex(RuntimeError, "formal status"):
                asset.validate()

    def test_proxy_matte_preserves_foreground_occluder(self):
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        base[..., 0] = 255
        depth = np.ones((2, 2), dtype=np.float32)
        mask = np.asarray(((True, False), (True, False)))
        alpha = np.full((2, 2, 1), 0.5, dtype=np.float32)
        premultiplied = np.zeros((2, 2, 3), dtype=np.float32)
        premultiplied[..., 2] = 0.5
        frame = AppearanceFrame(premultiplied, alpha, 1.0, "fake")
        result = composite_with_proxy_depth(base, depth, mask, frame)
        np.testing.assert_array_equal(result.rgb_u8[0, 0], (128, 0, 128))
        np.testing.assert_array_equal(result.rgb_u8[0, 1], (255, 0, 0))
        self.assertAlmostEqual(result.visible_proxy_fraction, 0.5)
        self.assertAlmostEqual(result.gaussian_alpha_clipped_fraction, 0.5)

    def test_safe_binding_latches_renderer_failure(self):
        class BrokenRenderer:
            backend = "broken"

            def __init__(self):
                self.calls = 0
                self.closed = False

            def render(self, _camera):
                self.calls += 1
                raise RuntimeError("render failed")

            def close(self):
                self.closed = True

        renderer = BrokenRenderer()
        binding = SafeAppearanceBinding(renderer)
        camera = PinholeCamera(2, 2, np.eye(3), np.eye(4))
        first = binding.render(camera)
        second = binding.render(camera)
        self.assertIsNone(first.frame)
        self.assertIsNone(second.frame)
        self.assertIn("render failed", first.error)
        self.assertEqual(renderer.calls, 1)
        self.assertTrue(renderer.closed)
        self.assertEqual(binding.metrics()["failures"], 1)


if __name__ == "__main__":
    unittest.main()
