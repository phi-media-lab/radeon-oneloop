import unittest
from pathlib import Path


class SevaInstallContractTests(unittest.TestCase):
    def test_installer_uses_interactive_credential_without_token_argument(self):
        value = Path("ops/install_phi_seva_model.sh").read_text()
        self.assertIn("HfApi().whoami()", value)
        self.assertIn("expected_hf_user=fbsh96", value)
        self.assertIn('actual_hf_user != expected_hf_user', value)
        self.assertIn('"huggingface_user": hf_user', value)
        self.assertIn("e538e251c1009e9a41cf8b7fee5f21332a1960de", value)
        self.assertIn("modelv1.1.safetensors", value)
        self.assertIn('"credential_material_recorded": False', value)
        self.assertNotIn("--token", value)
        self.assertNotIn("HF_TOKEN", value)
        self.assertIn("hashes.sha256", value)

    def test_pipeline_stops_at_human_review(self):
        value = Path("ops/run_phi_seva_until_review.sh").read_text()
        self.assertIn("REVIEW_REQUIRED.json", value)
        self.assertIn('"automatic_promotion": False', value)
        self.assertNotIn("run_phi_seva_pseudoview_colmap.sh", value)
        self.assertIn("hashes.sha256", value)

    def test_generation_and_audit_run_ids_can_be_prebound(self):
        generation = Path("ops/run_phi_seva_four_view.sh").read_text()
        audit = Path("ops/run_phi_seva_orbit_audit.sh").read_text()
        self.assertIn("ONELOOP_SEVA_RUN_ID", generation)
        self.assertIn("ONELOOP_SEVA_AUDIT_RUN_ID", audit)
        self.assertIn("[a-zA-Z0-9._-]", generation)
        self.assertIn("[a-zA-Z0-9._-]", audit)
        self.assertIn("hashes.sha256", generation)

    def test_post_review_runner_cannot_bypass_acceptance(self):
        value = Path("ops/run_phi_seva_after_review.sh").read_text()
        self.assertIn('review.get("decision") != "accepted_low_confidence_pseudoviews"', value)
        self.assertIn('request.get("automatic_promotion") is not False', value)
        self.assertIn('review.get("evidence", {}).get("audit_metrics_sha256")', value)
        self.assertIn("run_phi_seva_pseudoview_colmap.sh", value)


if __name__ == "__main__":
    unittest.main()
