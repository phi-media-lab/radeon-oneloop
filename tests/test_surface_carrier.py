import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian.surface_carrier import (
    analytical_lateral_initialization,
    apply_xy_scale,
    metric_base_vertices,
    silhouette_iou,
    write_colored_ply,
)


class SurfaceCarrierTests(unittest.TestCase):
    def test_metric_base_locks_complete_height_without_translation(self):
        vertices = np.asarray(
            [(-0.04, -0.03, -0.04), (0.04, 0.03, 0.06)], dtype=np.float64
        )
        scaled, report = metric_base_vertices(vertices, object_height_m=0.095)
        self.assertAlmostEqual(float(np.ptp(scaled, axis=0)[2]), 0.095, places=12)
        np.testing.assert_allclose(scaled, vertices * 0.95)
        self.assertAlmostEqual(report["uniform_metric_scale"], 0.95)

    def test_lateral_initialization_uses_opposite_view_medians(self):
        vertices = np.asarray(
            [(-0.04, -0.05, -0.0475), (0.04, 0.05, 0.0475)], dtype=np.float64
        )
        cameras = []
        for label, width in (("front", 900), ("rear", 800), ("right", 700), ("left", 600)):
            cameras.append(
                {
                    "view_label": label,
                    "foreground_bbox_xyxy": [0, 0, width, 950],
                }
            )
        scale, report = analytical_lateral_initialization(
            cameras, vertices, object_height_m=0.095
        )
        self.assertAlmostEqual(report["target_x_m"], 0.085)
        self.assertAlmostEqual(report["target_y_m"], 0.065)
        np.testing.assert_allclose(scale, [0.085 / 0.08, 0.065 / 0.10])
        final = apply_xy_scale(vertices, scale)
        np.testing.assert_allclose(np.ptp(final, axis=0)[:2], [0.085, 0.065])

    def test_silhouette_iou_handles_overlap_and_empty_masks(self):
        a = np.asarray(((1, 1), (0, 0)), dtype=bool)
        b = np.asarray(((1, 0), (1, 0)), dtype=bool)
        self.assertAlmostEqual(silhouette_iou(a, b), 1.0 / 3.0)
        self.assertEqual(silhouette_iou(np.zeros_like(a), np.zeros_like(a)), 1.0)

    def test_colored_ply_preserves_confidence_and_source_count_fields(self):
        vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        colors = np.asarray(((1, 2, 3), (4, 5, 6), (7, 8, 9)), dtype=np.uint8)
        confidence = np.asarray((0.1, 0.5, 1.0), dtype=np.float32)
        source_count = np.asarray((0, 1, 2), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "carrier.ply"
            write_colored_ply(path, vertices, faces, colors, confidence, source_count)
            text = path.read_text(encoding="ascii")
        self.assertIn("property float confidence", text)
        self.assertIn("property uchar source_count", text)
        self.assertIn("element face 1", text)


if __name__ == "__main__":
    unittest.main()
