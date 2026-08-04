import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian.prepare_four_view_generation import (
    FourViewInputError,
    orbit_c2w,
    prepare_generation_input,
    sha256_file,
    validate_generation_input,
)


CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


def write_png(path: Path, value: int, channels: int = 3) -> None:
    import cv2

    shape = (32, 32) if channels == 1 else (32, 32, channels)
    image = np.full(shape, value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write fixture: {path}")


def build_reviewed_root(root: Path) -> Path:
    reviewed = root / "reviewed"
    labels = (
        ("anchor_front", 0),
        ("anchor_right", -90),
        ("anchor_rear", 180),
        ("anchor_left", 90),
    )
    views = []
    for index, (view_id, azimuth) in enumerate(labels):
        files = {}
        for kind, value, channels in (
            ("image", 50 + index, 3),
            ("neutral_image", 70 + index, 3),
            ("hard_mask", 255, 1),
            ("soft_alpha", 255, 1),
        ):
            path = reviewed / "01_normalized" / kind / f"{view_id}.png"
            write_png(path, value, channels)
            files[kind] = {
                "relpath": path.relative_to(reviewed).as_posix(),
                "sha256": sha256_file(path),
            }
        views.append(
            {
                "id": view_id,
                "instance_id": "same_unit",
                "provenance": "observed",
                "tier": "A",
                "prepared": True,
                "mask_status": "reviewed_pass",
                "roles": ["pose", "photometric", "identity", "generation_input"],
                "nominal_camera_orbit_deg": {"azimuth": azimuth, "elevation": 0},
                **files,
            }
        )
    manifest = {
        "schema_version": "radeon_oneloop.object_asset_manifest.v1",
        "asset_name": "fixture",
        "formal": False,
        "redistribution": False,
        "coordinate_convention": {
            "front_axis": "+Y",
            "up_axis": "+Z",
            "viewer_left_axis": "+X",
            "unit": "m",
            "origin": "plush_body_center",
        },
        "metric_anchor": {"dimension": "overall_height", "value_m": 0.095},
        "summary": {"prepared_count": 4, "generated_count": 0},
        "views": views,
    }
    manifest_path = reviewed / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    hash_lines = []
    for path in sorted(reviewed.rglob("*.png")) + [manifest_path]:
        hash_lines.append(f"{sha256_file(path)}  {path.relative_to(reviewed).as_posix()}")
    (reviewed / "hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    (reviewed / "DONE").write_text(
        json.dumps(
            {
                "schema_version": "radeon_oneloop.object_asset_stage_done.v1",
                "manifest_sha256": sha256_file(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return reviewed


@unittest.skipUnless(CV2_AVAILABLE, "OpenCV is required")
class PrepareFourViewGenerationTests(unittest.TestCase):
    def test_builds_geometry_free_seva_and_hunyuan_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prepared"
            result = prepare_generation_input(build_reviewed_root(root), output)
            self.assertIsNone(result["source_policy"]["geometry_input"])
            self.assertFalse(result["source_policy"]["surface_carrier_allowed"])
            self.assertEqual(len(result["observed_inputs"]), 4)
            split = json.loads(
                (
                    output
                    / "seva"
                    / "graffiti_mickey_four_view"
                    / "train_test_split_4.json"
                ).read_text()
            )
            self.assertEqual(split["train_ids"], [0, 1, 2, 3])
            self.assertEqual(split["test_ids"], list(range(4, 53)))
            cameras = np.load(output / "target_cameras.npz")
            self.assertEqual(cameras["cam_c2w"].shape, (49, 4, 4))
            self.assertAlmostEqual(float(cameras["azimuth_deg"][0]), 0.0)
            self.assertAlmostEqual(float(cameras["azimuth_deg"][-1]), 360.0 * 48.0 / 49.0)
            self.assertFalse(result["target_orbit"]["endpoint_duplicate"])
            self.assertEqual(result["target_orbit"]["path_topology"], "cyclic")
            for name in ("front.png", "right.png", "back.png", "left.png"):
                self.assertTrue((output / "hunyuan3d_2mv" / name).is_file())

    def test_rejects_generated_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviewed = build_reviewed_root(root)
            manifest_path = reviewed / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["summary"]["generated_count"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(FourViewInputError, "does not bind manifest"):
                prepare_generation_input(reviewed, root / "prepared")

    def test_output_hash_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prepared"
            prepare_generation_input(build_reviewed_root(root), output)
            target = output / "target_schedule.json"
            target.write_text(target.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(FourViewInputError, "hash mismatch"):
                validate_generation_input(output)



class OrbitCameraTests(unittest.TestCase):
    def test_front_camera_is_on_positive_y_and_looks_at_origin(self):
        matrix = orbit_c2w(0.0, 0.0, 2.0)
        np.testing.assert_allclose(matrix[:3, 3], [0.0, 2.0, 0.0], atol=1e-8)
        forward = -matrix[:3, 2]
        np.testing.assert_allclose(forward, [0.0, -1.0, 0.0], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
