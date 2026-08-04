import importlib.util
import unittest

import numpy as np

from gaussian.texture_learned_mesh_four_views import (
    _render_perspective,
    canonical_orbit_extrinsic,
    _view_basis_from_azimuth,
    nearest_orbit_index,
    vista4d_unique_azimuths,
    vista4d_camera_track,
)


class LearnedMeshTextureTests(unittest.TestCase):
    def test_front_orbit_basis_is_orthonormal(self):
        right, up, forward = _view_basis_from_azimuth(0.0)
        basis = np.stack((right, up, forward), axis=0)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(forward, [0.0, -1.0, 0.0], atol=1e-8)

    def test_closed_orbit_basis_matches(self):
        for first, last in zip(
            _view_basis_from_azimuth(0.0), _view_basis_from_azimuth(360.0), strict=True
        ):
            np.testing.assert_allclose(first, last, atol=1e-8)

    def test_vista4d_schedule_has_no_duplicate_endpoint(self):
        azimuths = vista4d_unique_azimuths()
        self.assertEqual(len(azimuths), 49)
        self.assertEqual(float(azimuths[0]), 0.0)
        self.assertLess(float(azimuths[-1]), 360.0)
        self.assertAlmostEqual(float(azimuths[-1]), 360.0 * 48.0 / 49.0)
        self.assertEqual(nearest_orbit_index(azimuths, 0.0), 0)
        self.assertEqual(nearest_orbit_index(azimuths, 90.0), 12)
        self.assertEqual(nearest_orbit_index(azimuths, 180.0), 24)
        self.assertEqual(nearest_orbit_index(azimuths, 270.0), 37)

    def test_perspective_render_keeps_fixed_metric_scale(self):
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is required")
        vertices = np.asarray(
            [[-0.01, 0.0, -0.01], [0.01, 0.0, -0.01], [0.0, 0.0, 0.01]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        colors = np.asarray([[255, 0, 0]] * 3, dtype=np.uint8)
        camera = np.eye(4)
        camera[2, 3] = 0.2
        intrinsic = np.asarray([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0, 0, 1]])
        image, alpha = _render_perspective(
            vertices,
            faces,
            colors,
            camera_from_object=camera,
            intrinsic=intrinsic,
            size_wh=(64, 64),
        )
        self.assertEqual(image.shape, (64, 64, 3))
        self.assertGreater(int(np.count_nonzero(alpha)), 0)

    def test_vista_camera_track_uses_same_unique_azimuths(self):
        intrinsic = np.asarray([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0, 0, 1]])
        c2w, packed = vista4d_camera_track(
            frames=49, intrinsic_3x3=intrinsic, distance_m=0.24
        )
        self.assertEqual(c2w.shape, (49, 4, 4))
        self.assertEqual(packed.shape, (49, 4))
        first_w2c = canonical_orbit_extrinsic(0.0, distance_m=0.24)
        conversion = np.diag([-1.0, -1.0, 1.0, 1.0])
        np.testing.assert_allclose(c2w[0], conversion @ np.linalg.inv(first_w2c))


if __name__ == "__main__":
    unittest.main()
