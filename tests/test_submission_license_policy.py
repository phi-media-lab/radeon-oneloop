from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from submission.validate_release import validate_restricted_artifacts


class SubmissionLicensePolicyTests(unittest.TestCase):
    def test_seva_and_known_outputs_are_excluded_without_clearance(self):
        policy = json.loads(Path("submission/license_policy.json").read_text())
        self.assertEqual(
            policy["schema_version"],
            "radeon_oneloop.submission_license_policy.v1",
        )
        record = policy["restricted_artifacts"][0]
        self.assertEqual(record["model"], "stabilityai/stable-virtual-camera")
        self.assertFalse(record["competition_submission_clearance"])
        self.assertIn(
            "ad538d0f1d4da96293aed7de5f9f33030435870c1c4339187f48c9dfa25bb4f2",
            record["known_descendant_sha256"],
        )

    def test_release_scan_rejects_exact_restricted_descendant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "submission").mkdir()
            (root / "artifacts/formal").mkdir(parents=True)
            # Bind a synthetic payload through a copied policy so this test
            # does not need any private or generated media.
            payload = root / "artifacts/formal/restricted.bin"
            payload.write_bytes(b"restricted-test-payload")
            import hashlib

            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            policy = {
                "schema_version": "radeon_oneloop.submission_license_policy.v1",
                "restricted_artifacts": [
                    {
                        "competition_submission_clearance": False,
                        "known_descendant_sha256": [digest],
                    }
                ],
            }
            (root / "submission/license_policy.json").write_text(
                json.dumps(policy) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "restricted noncommercial"):
                validate_restricted_artifacts(root)


if __name__ == "__main__":
    unittest.main()
