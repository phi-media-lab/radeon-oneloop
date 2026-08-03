import json
from pathlib import Path
import unittest

try:
    import yaml
except ImportError:  # The no-install scaffold check still validates YAML with Ruby.
    yaml = None


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class FormalRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.registry = yaml.safe_load(
            (cls.root / "ops/formal_run_registry.yaml").read_text(encoding="utf-8")
        )

    def test_dataset_registry_matches_published_build_manifest(self) -> None:
        manifest = json.loads(
            (self.root / "artifacts/formal/dataset/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        dataset = self.registry["formal_dataset"]
        self.assertEqual(dataset["builder_commit"], manifest["git_commit"])
        self.assertEqual(dataset["dataset_sha256"], manifest["dataset_hash"])
        self.assertEqual(
            dataset["phase_targets_sha256"], manifest["phase_targets_sha256"]
        )

    def test_registered_job_fields_match_public_manifests(self) -> None:
        for run in self.registry["runs"]:
            with self.subTest(job_id=run["id"]):
                evidence = self.root / run["evidence"]
                self.assertTrue((evidence / "DONE").is_file())
                manifest = json.loads(
                    (evidence / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(run["id"], manifest["job_id"])
                self.assertEqual(run["status"], manifest["status"])
                self.assertEqual(run["host"], manifest["host"])
                self.assertEqual(run["gpu_uid"], manifest["gpu_uid"])
                self.assertEqual(run["git_commit"], manifest["git_commit"])
                self.assertEqual(run["config_sha256"], manifest["config_hash"])
                self.assertEqual(run["dataset_sha256"], manifest["dataset_hash"])
                if "parent_checkpoint_sha256" in run:
                    self.assertEqual(
                        run["parent_checkpoint_sha256"],
                        manifest["parent_checkpoint"],
                    )


if __name__ == "__main__":
    unittest.main()
