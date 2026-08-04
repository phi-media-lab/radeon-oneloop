import unittest
from pathlib import Path
import tempfile

import numpy as np

from gaussian.audit_vista4d_completion import (
    inferred_foreground,
    make_anchor_comparison,
    make_contact_sheet,
    mask_iou,
    masked_mae,
    write_binary_masks,
)


class Vista4DCompletionAuditTests(unittest.TestCase):
    def test_masked_mae_ignores_pixels_outside_support(self):
        lhs = np.zeros((2, 2, 3), dtype=np.uint8)
        rhs = np.zeros_like(lhs)
        rhs[0, 0] = 255
        rhs[1, 1] = 255
        mask = np.asarray(((True, False), (False, False)))
        self.assertAlmostEqual(masked_mae(lhs, rhs, mask), 1.0)

    def test_contact_sheet_uses_requested_nine_frames(self):
        frames = np.stack(
            [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(49)]
        )
        sheet = make_contact_sheet(frames, (0, 6, 12, 18, 24, 30, 36, 42, 48))
        self.assertEqual(sheet.shape, (6, 9, 3))
        self.assertEqual(int(sheet[0, 0, 0]), 0)
        self.assertEqual(int(sheet[-1, -1, 0]), 48)

    def test_anchor_comparison_has_source_generated_and_difference_rows(self):
        source = np.zeros((49, 2, 3, 3), dtype=np.uint8)
        generated = np.full_like(source, 10)
        sheet = make_anchor_comparison(source, generated, (0, 12, 24, 37))
        self.assertEqual(sheet.shape, (6, 12, 3))
        self.assertTrue(np.all(sheet[:2] == 0))
        self.assertTrue(np.all(sheet[2:4] == 10))
        self.assertTrue(np.all(sheet[4:] == 30))

    def test_mask_iou_handles_partial_overlap_and_empty_union(self):
        lhs = np.asarray(((True, True), (False, False)))
        rhs = np.asarray(((True, False), (True, False)))
        self.assertAlmostEqual(mask_iou(lhs, rhs), 1.0 / 3.0)
        self.assertEqual(mask_iou(np.zeros_like(lhs), np.zeros_like(lhs)), 1.0)

    def test_foreground_inference_fills_main_component(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed")
        frame = np.full((64, 64, 3), 250, dtype=np.uint8)
        frame[18:46, 20:44] = (80, 90, 100)
        frame[26:38, 28:36] = 250
        frame[2:4, 2:4] = 0
        mask = inferred_foreground(frame)
        self.assertTrue(mask[32, 32])
        self.assertTrue(mask[20, 22])
        self.assertFalse(mask[2, 2])

    def test_generated_mask_bundle_is_binary_and_complete(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        masks = np.zeros((49, 4, 5), dtype=bool)
        masks[:, 1:3, 2:4] = True
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "generated_masks"
            write_binary_masks(root, masks)
            paths = sorted(root.glob("*.png"))
            self.assertEqual(len(paths), 49)
            decoded = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(set(np.unique(decoded).tolist()), {0, 255})


if __name__ == "__main__":
    unittest.main()
