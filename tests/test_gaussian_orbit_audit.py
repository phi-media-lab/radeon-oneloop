import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from sim.genesis_so101.gaussian_orbit_audit import (
    canonical_orbit_extrinsic,
    scaled_intrinsic,
)
from sim.genesis_so101.gaussian_appearance import nonformal_candidate_asset


class GaussianOrbitAuditTests(unittest.TestCase):
    def test_orbit_matches_pinned_anchor_camera_order(self):
        expected = {
            0.0: np.asarray(
                ((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0.3), (0, 0, 0, 1))
            ),
            90.0: np.asarray(
                ((0, -1, 0, 0), (0, 0, -1, 0), (1, 0, 0, 0.3), (0, 0, 0, 1))
            ),
            180.0: np.asarray(
                ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0.3), (0, 0, 0, 1))
            ),
            270.0: np.asarray(
                ((0, 1, 0, 0), (0, 0, -1, 0), (-1, 0, 0, 0.3), (0, 0, 0, 1))
            ),
        }
        for angle, matrix in expected.items():
            np.testing.assert_allclose(canonical_orbit_extrinsic(angle), matrix, atol=1e-12)

    def test_orbit_closes_at_360_degrees(self):
        np.testing.assert_allclose(
            canonical_orbit_extrinsic(360.0),
            canonical_orbit_extrinsic(0.0),
            atol=1e-12,
        )

    def test_intrinsic_scaling_preserves_normalized_projection(self):
        intrinsic = np.asarray(((1000.0, 0.0, 511.5), (0.0, 1000.0, 511.5), (0, 0, 1)))
        scaled = scaled_intrinsic(intrinsic, (1024, 1024), (512, 256))
        np.testing.assert_allclose(
            scaled,
            ((500.0, 0.0, 255.75), (0.0, 250.0, 127.875), (0.0, 0.0, 1.0)),
        )

    def test_invalid_orbit_distance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            canonical_orbit_extrinsic(0.0, distance_m=0.0)

    def test_candidate_loader_requires_nonformal_self_bound_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "appearance_observed_canonical.ply"
            cameras = root / "cameras_observed.json"
            provenance = root / "provenance.json"
            ply.write_bytes(b"candidate")
            cameras.write_text(
                json.dumps({"camera_model": "PINHOLE_OPENCV", "cameras": []}),
                encoding="utf-8",
            )
            ply_sha = hashlib.sha256(ply.read_bytes()).hexdigest()
            provenance.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.observed_core_canonicalization.v1",
                        "formal": False,
                        "eligible_for_heldout_real_metrics": False,
                        "output_ply_sha256": ply_sha,
                        "gaussian_count": 1,
                        "observed_only_training": True,
                        "provenance_class": "observed_core_candidate",
                    }
                ),
                encoding="utf-8",
            )
            asset = nonformal_candidate_asset(root)
            self.assertFalse(asset.validate()["formal"])
            value = json.loads(provenance.read_text())
            value["formal"] = True
            provenance.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "formal=false"):
                nonformal_candidate_asset(root)


if __name__ == "__main__":
    unittest.main()
