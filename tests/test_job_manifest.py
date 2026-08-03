import json
import unittest
from pathlib import Path


class JobManifestSchemaTests(unittest.TestCase):
    def test_runner_roles_are_accepted_by_schema(self):
        schema = json.loads(Path("ops/job_manifest.schema.json").read_text())
        roles = set(schema["properties"]["role"]["enum"])
        self.assertTrue({"genesis_smoke", "act_smoke", "act_train", "act_eval"} <= roles)

    def test_parent_checkpoint_is_a_sha256_when_present(self):
        schema = json.loads(Path("ops/job_manifest.schema.json").read_text())
        self.assertEqual(
            schema["properties"]["parent_checkpoint"]["pattern"],
            "^[0-9a-f]{64}$",
        )


if __name__ == "__main__":
    unittest.main()
