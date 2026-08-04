import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gaussian.record_generated_render_audit import main


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(root: Path, ply_hash: str) -> None:
    root.mkdir()
    manifest = {"ply_sha256": ply_hash, "gaussian_count": 10}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "DONE").write_text("{}")
    (root / "hashes.sha256").write_text(f"{sha256(root / 'manifest.json')}  manifest.json\n")


class GeneratedRenderAuditTests(unittest.TestCase):
    def test_records_rejected_direct_fill_without_rejecting_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            baseline = root / "baseline"
            make_run(candidate, "candidate-ply")
            make_run(baseline, "baseline-ply")
            provenance = root / "provenance.json"
            provenance.write_text(
                json.dumps(
                    {"output_ply_sha256": "candidate-ply", "eligible_for_formal_metrics": False}
                )
            )
            montage = root / "montage.jpg"
            montage.write_bytes(b"jpg")
            output = root / "audit"
            argv = [
                "record_generated_render_audit.py",
                "--render-run", str(candidate),
                "--baseline-render-run", str(baseline),
                "--source-provenance", str(provenance),
                "--montage", str(montage),
                "--output", str(output),
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            audit = json.loads((output / "audit_manifest.json").read_text())
            self.assertFalse(audit["review"]["direct_appearance_fill_accepted"])
            self.assertFalse(audit["review"]["accepted_for_confidence_pruning"])
            self.assertTrue(audit["review"]["appearance_pseudoview_branch_retained"])


if __name__ == "__main__":
    unittest.main()
