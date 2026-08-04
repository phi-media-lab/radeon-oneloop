from __future__ import annotations

import numpy as np
import unittest

from gaussian.audit_seva_orbit import centroid, infer_foreground, mask_iou, masked_rgb_mae


class SevaOrbitAuditTests(unittest.TestCase):
    def test_foreground_inference_and_basic_metrics(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed")
        frame = np.full((96, 96, 3), 245, dtype=np.uint8)
        frame[24:72, 30:66] = np.array([80, 120, 180], dtype=np.uint8)
        inferred = infer_foreground(frame)
        expected = np.zeros((96, 96), dtype=bool)
        expected[24:72, 30:66] = True
        self.assertGreater(mask_iou(inferred, expected), 0.95)
        self.assertEqual(masked_rgb_mae(frame, frame.copy(), expected), 0.0)
        cx, cy = centroid(inferred)
        self.assertAlmostEqual(cx, 0.5, delta=0.02)
        self.assertAlmostEqual(cy, 0.5, delta=0.02)

    def test_mask_iou_validates_shape_and_empty_union(self) -> None:
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(mask_iou(empty, empty), 1.0)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            mask_iou(empty, np.zeros((5, 5), dtype=bool))

    def test_masked_rgb_mae_rejects_empty_support(self) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            masked_rgb_mae(frame, frame, np.zeros((4, 4), dtype=bool))


if __name__ == "__main__":
    unittest.main()
