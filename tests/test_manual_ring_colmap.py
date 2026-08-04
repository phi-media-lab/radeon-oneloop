import unittest

import numpy as np

from gaussian.manual_ring_colmap import six_neighbour_boundary


class ManualRingColmapTests(unittest.TestCase):
    def test_six_neighbour_boundary_removes_only_cube_interior(self) -> None:
        occupied = np.ones((5, 5, 5), dtype=bool)
        boundary = six_neighbour_boundary(occupied)
        self.assertEqual(int(boundary.sum()), 5**3 - 3**3)
        self.assertFalse(boundary[2, 2, 2])
        self.assertTrue(boundary[0, 2, 2])

    def test_isolated_voxel_is_surface(self) -> None:
        occupied = np.zeros((5, 5, 5), dtype=bool)
        occupied[2, 2, 2] = True
        boundary = six_neighbour_boundary(occupied)
        np.testing.assert_array_equal(boundary, occupied)

    def test_rejects_non_volume(self) -> None:
        with self.assertRaisesRegex(ValueError, "3-D"):
            six_neighbour_boundary(np.ones((5, 5), dtype=bool))


if __name__ == "__main__":
    unittest.main()
