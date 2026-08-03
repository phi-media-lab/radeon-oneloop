import json
import tempfile
import unittest
from pathlib import Path

from gaussian.vksplat_train import validate_dataset


class GaussianContractTests(unittest.TestCase):
    def test_colmap_dataset_fingerprint_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "sparse/0").mkdir(parents=True)
            for index in range(8):
                (root / "images" / f"{index:03d}.png").write_bytes(bytes([index]))
            for name in ("cameras", "images", "points3D"):
                (root / "sparse/0" / f"{name}.bin").write_bytes(name.encode())
            first = validate_dataset(root, "images", "sparse/0")
            second = validate_dataset(root, "images", "sparse/0")
            self.assertEqual(first, second)
            self.assertEqual(first["images"], 8)
            self.assertEqual(first["model_format"], "bin")

    def test_capture_schema_is_valid_json(self):
        schema = json.loads(Path("gaussian/workspace_capture.schema.json").read_text())
        self.assertIn("scale_anchor", schema["required"])
