import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian.hunyuan3d_mv_generate import mesh_statistics, validate_local_snapshot


class Hunyuan3DMeshStatisticsTests(unittest.TestCase):
    def test_records_valid_tetrahedron(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
        result = mesh_statistics(vertices, faces)
        self.assertEqual(result["vertices"], 4)
        self.assertEqual(result["triangles"], 4)
        self.assertEqual(result["unique_triangles"], 4)
        self.assertEqual(result["max_extent_raw"], 1.0)

    def test_rejects_nonfinite_vertex(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, np.nan]], dtype=np.float32
        )
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mesh_statistics(vertices, faces)

    def test_rejects_out_of_bounds_face(self):
        vertices = np.eye(4, 3, dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 4]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            mesh_statistics(vertices, faces)

    def test_local_snapshot_requires_content_addressed_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main"
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "40-hex"):
                validate_local_snapshot(path, "hunyuan3d-dit-v2-mv")


if __name__ == "__main__":
    unittest.main()
