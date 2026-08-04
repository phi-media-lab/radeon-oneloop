import unittest

import numpy as np

from gaussian.align_generated_mesh_four_views import (
    metric_transform,
    signed_permutation_rotations,
    silhouette_iou,
)


class GeneratedMeshAlignmentTests(unittest.TestCase):
    def test_rotation_search_is_exactly_24_right_handed_bases(self):
        rotations = signed_permutation_rotations()
        self.assertEqual(len(rotations), 24)
        for matrix in rotations:
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0)

    def test_metric_transform_sets_canonical_height(self):
        vertices = np.array(
            [[-2, -1, -3], [2, -1, -3], [-2, 1, 3], [2, 1, 3]], dtype=np.float64
        )
        transformed, matrix = metric_transform(vertices, np.eye(3), 0.095)
        self.assertAlmostEqual(float(np.ptp(transformed[:, 2])), 0.095)
        self.assertEqual(matrix.shape, (4, 4))
        np.testing.assert_allclose(
            0.5 * (transformed.min(axis=0) + transformed.max(axis=0)),
            np.zeros(3),
            atol=1e-10,
        )

    def test_silhouette_iou(self):
        a = np.array([[True, True], [False, False]])
        b = np.array([[True, False], [True, False]])
        self.assertAlmostEqual(silhouette_iou(a, b), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
