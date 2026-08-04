import unittest

import numpy as np

from sim.genesis_so101.gaussian_genesis_static import (
    gaussian_center_extents,
    gaussian_support_extents,
)


class GaussianGenesisStaticTests(unittest.TestCase):
    def test_metric_extent_preserves_boundary_splats_but_rejects_export_outliers(self):
        body_z = np.linspace(-0.0475, 0.0475, 29_994, dtype=np.float64)
        body = np.column_stack((np.zeros_like(body_z), np.zeros_like(body_z), body_z))
        outliers = np.asarray(
            [[0.0, 0.0, -1.0]] * 3 + [[0.0, 0.0, 1.0]] * 3,
            dtype=np.float64,
        )
        full, trimmed = gaussian_center_extents(np.vstack((outliers, body)))
        self.assertAlmostEqual(float(full[2]), 2.0)
        self.assertAlmostEqual(float(trimmed[2]), 0.095, delta=0.00025)

    def test_metric_extent_rejects_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "finite Nx3"):
            gaussian_center_extents(np.asarray([[0.0, np.nan, 0.0]]))

    def test_support_extent_includes_anisotropic_gaussian_size(self):
        centers = np.asarray(((0.0, 0.0, -0.0455), (0.0, 0.0, 0.0455)))
        scales = np.asarray(((0.001, 0.001, 0.001),) * 2)
        rotations = np.asarray(((1.0, 0.0, 0.0, 0.0),) * 2)

        full, robust = gaussian_support_extents(
            centers, scales, rotations, trim_percent=0.0
        )

        self.assertAlmostEqual(float(full[2]), 0.095)
        self.assertAlmostEqual(float(robust[2]), 0.095)


if __name__ == "__main__":
    unittest.main()
