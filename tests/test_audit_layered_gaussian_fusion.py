import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from gaussian.audit_layered_gaussian_fusion import FRAME_COUNT, audit


class LayeredFusionAuditTest(unittest.TestCase):
    def test_accepts_small_nonformal_overlay(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            observed = root / "observed"
            fused = root / "fused"
            observed.mkdir()
            fused.mkdir()
            (observed / "render_manifest.json").write_text("{}\n")
            (fused / "render_manifest.json").write_text("{}\n")
            source = np.ones((32, 32, 3), dtype=np.float32)
            source[8:24, 8:24] = 100 / 255
            candidate = source.copy()
            candidate[15:17, 15:17] = 90 / 255
            source_orbit = np.repeat(source[None, ...], FRAME_COUNT, axis=0)
            candidate_orbit = np.repeat(candidate[None, ...], FRAME_COUNT, axis=0)

            with patch(
                "gaussian.audit_layered_gaussian_fusion._load_frames",
                side_effect=[source_orbit, candidate_orbit],
            ):
                result = audit(observed, fused, root / "audit")

            self.assertEqual(result["gates"]["observed_anchor_safety"]["status"], "pass")
            self.assertEqual(
                result["decision"], "accept_as_optional_nonformal_toggle_default_off"
            )
            self.assertFalse(result["formal"])
            self.assertTrue((root / "audit" / "DONE").is_file())
            persisted = json.loads((root / "audit" / "metrics.json").read_text())
            self.assertEqual(persisted["frame_count"], 49)


if __name__ == "__main__":
    unittest.main()
