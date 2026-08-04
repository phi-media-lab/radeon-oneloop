import unittest

import numpy as np

from gaussian.audit_vista4d_mask_alignment import binary_mask_alignment


class Vista4DMaskAlignmentTests(unittest.TestCase):
    def test_binary_alignment_partitions_union(self):
        source = np.asarray(((True, True), (False, False)))
        point = np.asarray(((True, False), (True, False)))
        result = binary_mask_alignment(source, point)
        self.assertAlmostEqual(result["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(result["source_only_fraction_of_union"], 1.0 / 3.0)
        self.assertAlmostEqual(result["point_only_fraction_of_union"], 1.0 / 3.0)

    def test_empty_masks_are_perfectly_aligned(self):
        empty = np.zeros((2, 2), dtype=bool)
        result = binary_mask_alignment(empty, empty)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["source_support_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
