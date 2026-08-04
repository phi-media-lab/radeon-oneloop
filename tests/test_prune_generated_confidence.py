import unittest

import numpy as np

from gaussian.prune_generated_confidence import compute_depth_prune_mask


class PruneGeneratedConfidenceTests(unittest.TestCase):
    def test_keeps_multi_source_and_only_consistent_single_source(self):
        positions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.004],
                [0.0, 0.0, 0.98],
                [0.0, 0.0, 1.05],
            ]
        )
        source_view = np.zeros(4, dtype=np.uint8)
        cross_source = np.array([2, 1, 1, 1], dtype=np.uint8)
        camera = {
            "world_to_camera_opencv_4x4": np.eye(4).tolist(),
            "intrinsic_3x3": np.eye(3).tolist(),
        }
        masks = np.full((4, 2, 2), 255, dtype=np.uint8)
        depth = np.ones((4, 2, 2), dtype=np.float64)
        keep, report = compute_depth_prune_mask(
            positions,
            source_view,
            cross_source,
            [camera] * 4,
            masks,
            depth,
            source_max_abs_depth_error_m=0.008,
            max_front_conflict_m=0.004,
        )
        np.testing.assert_array_equal(keep, [True, True, False, False])
        self.assertEqual(report["output_gaussians"], 2)


if __name__ == "__main__":
    unittest.main()
