import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gaussian.record_generated_fill_audit import main


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeneratedFillAuditTests(unittest.TestCase):
    def test_records_separate_appearance_and_geometry_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            manifest = {
                "model": "UniSHARP",
                "checkpoint_sha256": "checkpoint",
                "hardware": {"device": "MI300X"},
                "outputs": [
                    {
                        "view_id": "anchor_front",
                        "relpath": "inference/front/gaussians.ply",
                        "sha256": "ply",
                        "gaussian_count": 1179648,
                    }
                ],
            }
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "DONE").write_text("{}")
            (run / "hashes.sha256").write_text(f"{sha256(run / 'manifest.json')}  manifest.json\n")
            failure = root / "failure"
            failure.mkdir()
            (failure / "FAILED").write_text("{}")
            diagnostic = [{"view_id": "anchor_front", "numeric_gate_passed": False}]
            (failure / "stderr.log").write_text(
                "trace\n" + "one or more SHARP-family-to-VGGT alignment gates failed: "
                + json.dumps(diagnostic) + "\n"
            )
            montage = root / "montage.jpg"
            montage.write_bytes(b"jpeg")
            output = root / "audit"
            argv = [
                "record_generated_fill_audit.py",
                "--generator-run", str(run),
                "--output", str(output),
                "--montage", str(montage),
                "--metric-fit-failure", str(failure),
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            audit = json.loads((output / "audit_manifest.json").read_text())
            self.assertTrue(audit["review"]["appearance_proposal_accepted"])
            self.assertFalse(audit["review"]["metric_geometry_accepted"])
            self.assertEqual(audit["metric_fit_failures"][0]["alignment_diagnostic"], diagnostic)


if __name__ == "__main__":
    unittest.main()
