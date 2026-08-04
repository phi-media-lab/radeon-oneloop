import json
import tempfile
import unittest
from pathlib import Path

from gaussian.unisharp_infer_with_frames import build_pseudoview_document


class UniSharpInferWithFramesTests(unittest.TestCase):
    def test_cropped_intrinsics_and_local_camera_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("forward", "rotate"):
                frame_dir = root / f"{kind}_frames"
                frame_dir.mkdir()
                for index in range(2):
                    (frame_dir / f"frame_{index:03d}.png").write_bytes(b"png")
            metadata = {
                "camera_kind": "perspective",
                "camera_json_entry": {
                    "intrinsics": [846.0, 840.0, 384.0, 384.0],
                    "source_camera_view_id": "anchor_front",
                    "source_image_sha256": "input",
                },
                "width": 768,
                "height": 768,
                "output_crop_border_fraction": 0.05,
                "forward_distance_m": 0.04,
                "rotate_radius_m": 0.02,
            }
            result = build_pseudoview_document(metadata, root)
            self.assertEqual(result["views"][0]["image_size_wh"], [692, 692])
            self.assertEqual(result["views"][0]["intrinsic_3x3"][0][2], 346.0)
            self.assertEqual(result["views"][0]["camera_to_generator_world_4x4"][2][3], 0.02)
            rotate_zero = result["views"][2]
            self.assertAlmostEqual(rotate_zero["camera_to_generator_world_4x4"][1][3], 0.02)
            json.dumps(result)


if __name__ == "__main__":
    unittest.main()
