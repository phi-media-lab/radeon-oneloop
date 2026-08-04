import json
import tempfile
import unittest
from pathlib import Path

import yaml

from gaussian.development_evidence import summarize


PLY_SHA = "7f01c1e6d8253d7f15162e2cb51e18845676fa1015983266b7d356d9b21aa706"


def _write(path: Path, value: object = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class DevelopmentEvidenceTests(unittest.TestCase):
    def test_orbit_requires_formal_asset_and_human_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "orbit_run"
            _write(root / "DONE", b"")
            (root / "manifest.yaml").parent.mkdir(parents=True, exist_ok=True)
            (root / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "formal": False,
                        "host_role": "amd_apu_nonformal_visual_audit",
                        "physical_output": False,
                    }
                ),
                encoding="utf-8",
            )
            _write(root / "hashes.sha256")
            _write(root / "artifacts/orbit_contact_sheet.png")
            _write(root / "artifacts/orbit_360.mp4")
            _write(
                root / "artifacts/metrics.json",
                {
                    "accepted_numeric": True,
                    "formal": False,
                    "physical_output": False,
                    "asset": {"formal": True, "hashes": {"ply": PLY_SHA}},
                    "orbit": {
                        "frames_without_duplicate_endpoint": 72,
                        "cycle_closure_rgb_mae": 0.0,
                        "border_contact_frames": 0,
                        "alpha_support_fraction": {"min": 0.4, "max": 0.6},
                    },
                    "render_ms": {"mean": 25.0},
                },
            )
            with self.assertRaisesRegex(ValueError, "visual-review"):
                summarize(root, mode="orbit", expected_ply_sha256=PLY_SHA)
            result = summarize(
                root,
                mode="orbit",
                expected_ply_sha256=PLY_SHA,
                visual_review="accepted_with_sparse_view_limit",
            )
            self.assertTrue(result["accepted"])
            self.assertFalse(result["heldout_quality_claim"])

    def test_live_redacts_raw_actions_to_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "live_run"
            _write(root / "DONE", b"")
            (root / "manifest.yaml").parent.mkdir(parents=True, exist_ok=True)
            (root / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "formal": False,
                        "host_role": "amd_apu_nonformal_runtime_integration",
                        "physical_output": False,
                    }
                ),
                encoding="utf-8",
            )
            _write(root / "hashes.sha256")
            _write(root / "gate.json", {"accepted": True, "physical_output": False})
            _write(
                root / "consumer/metrics.json",
                {
                    "physical_output_commands": False,
                    "packets": {"accepted": 100},
                    "watchdog": {"events": 0},
                },
            )
            _write(root / "renderer/READY")
            _write(root / "renderer/live_gaussian_first.png")
            _write(root / "renderer/live_gaussian_final.png")
            _write(root / "renderer/live_gaussian.mp4")
            _write(
                root / "renderer/metrics.json",
                {
                    "accepted": True,
                    "physical_output": False,
                    "asset": {"formal": True, "hashes": {"ply": PLY_SHA}},
                    "appearance": {"fallback_frames": 0},
                    "render": {"effective_hz": 8.0},
                },
            )
            publisher = {
                "schema_version": "radeon_oneloop.leader_publisher.v1",
                "physical_output_commands": False,
                "action_range": {
                    "action_names": [f"joint_{index}" for index in range(12)],
                    "span": [float(index) for index in range(12)],
                    "samples": 100,
                    "capture_start_gated": False,
                    "capture_started": True,
                },
                "haptic_feedback": {"output_commands": 0},
            }
            (root / "publisher.log").write_text(
                "periodic sample omitted\n" + json.dumps(publisher, indent=2) + "\n",
                encoding="utf-8",
            )
            result = summarize(root, mode="live", expected_ply_sha256=PLY_SHA)
            self.assertEqual(result["publisher"]["span"], publisher["action_range"]["span"])
            self.assertNotIn("raw_actions", result["publisher"])
            self.assertFalse(result["task_success_evaluated"])

    def test_live_rejects_any_physical_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root / "DONE", b"")
            (root / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "formal": False,
                        "host_role": "amd_apu_nonformal_runtime_integration",
                        "physical_output": False,
                    }
                ),
                encoding="utf-8",
            )
            _write(root / "hashes.sha256")
            _write(root / "gate.json", {"accepted": True, "physical_output": False})
            _write(
                root / "consumer/metrics.json",
                {"physical_output_commands": True},
            )
            _write(root / "renderer/metrics.json", {})
            _write(
                root / "publisher.log",
                {
                    "schema_version": "radeon_oneloop.leader_publisher.v1",
                    "physical_output_commands": False,
                },
            )
            with self.assertRaisesRegex(ValueError, "control emitted output"):
                summarize(root, mode="live", expected_ply_sha256=PLY_SHA)


if __name__ == "__main__":
    unittest.main()
