import unittest
from pathlib import Path


class SevaInstallContractTests(unittest.TestCase):
    def test_installer_uses_interactive_credential_without_token_argument(self):
        value = Path("ops/install_phi_seva_model.sh").read_text()
        self.assertIn("HfApi().whoami()", value)
        self.assertIn("expected_hf_user=fbsh96", value)
        self.assertIn('actual_hf_user != expected_hf_user', value)
        self.assertIn('"huggingface_user": hf_user', value)
        self.assertIn("files_metadata=True", value)
        self.assertIn("resolved.stat().st_size != expected_size", value)
        self.assertIn('filename.endswith(".safetensors") and expected_size == 0', value)
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
        self.assertIn('! -f "$model_root/config.yaml"', value)
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
        self.assertIn("seva_official_vae_31f26fde.patch", generation)
        self.assertIn("stabilityai/sd-vae-ft-mse", generation)
        self.assertIn("31f26fdeee1355a5c34592e401dd41e45d25a493", generation)
        self.assertIn("vae_provenance.json", generation)
        self.assertIn("a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815", generation)
        self.assertIn("timeout --signal=TERM --kill-after=60 10800", generation)
        self.assertIn('"credential_material_recorded": False', generation)

    def test_post_review_runner_cannot_bypass_acceptance(self):
        value = Path("ops/run_phi_seva_after_review.sh").read_text()
        self.assertIn('review.get("decision") != "accepted_low_confidence_pseudoviews"', value)
        self.assertIn('request.get("automatic_promotion") is not False', value)
        self.assertIn('review.get("evidence", {}).get("audit_metrics_sha256")', value)
        self.assertIn('review.get("evidence", {}).get("four_view_manifest_sha256")', value)
        self.assertIn('request["generation_run_id"]', value)
        self.assertIn("recovered SEVA review requires its explicit four-view input", value)
        self.assertIn("run_phi_seva_pseudoview_colmap.sh", value)

    def test_recovery_reuses_only_completed_inference_and_still_stops_for_review(self):
        value = Path("ops/recover_phi_seva_after_record_failure.sh").read_text()
        self.assertIn('[[ -f "$source_generation/FAILED"', value)
        self.assertIn('[[ ! -e "$failed_pipeline/DONE" ]]', value)
        self.assertIn('"inference_rerun": False', value)
        self.assertIn('"automatic_promotion": False', value)
        self.assertIn('"four_view_input": str(four_view.resolve())', value)
        self.assertIn("REVIEW_REQUIRED.json", value)
        self.assertNotIn("demo.py", value)

    def test_pseudoview_wrapper_preserves_early_failures(self):
        value = Path("ops/run_phi_seva_pseudoview_colmap.sh").read_text()
        self.assertIn("WRAPPER_FAILED.json", value)
        self.assertIn('failed_dir="$run_dir.FAILED"', value)
        self.assertIn("hashes.sha256", value)


if __name__ == "__main__":
    unittest.main()
