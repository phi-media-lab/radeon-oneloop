import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gaussian.record_object_visual_audit import inspect_run, sha256_file


def write_hashes(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root)}\n")
    (root / "hashes.sha256").write_text("".join(lines), encoding="utf-8")


class ObjectVisualAuditTests(unittest.TestCase):
    def make_run(self, root: Path, view: str) -> Path:
        train = root / "train"
        train.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps({"dataset_manifest_sha256": "a" * 64}), encoding="utf-8"
        )
        (train / "val_00000.png").write_bytes(b"render")
        (train / "train.json").write_text(
            json.dumps(
                {
                    "val_images": [
                        {"image_path": f"/private/000_eval_probe_anchor_{view}.png"}
                    ],
                    "train_images": [
                        {"image_path": f"/private/anchor_{name}.png"}
                        for name in ("front", "left", "rear", "right")
                    ],
                }
            ),
            encoding="utf-8",
        )
        write_hashes(root)
        (root / "DONE").touch()
        return root

    def test_inspect_run_accepts_exact_probe_and_four_observed_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_run(Path(temporary), "rear")
            result = inspect_run("rear", root)
            self.assertEqual(result["view"], "rear")
            self.assertEqual(result["dataset_manifest_sha256"], "a" * 64)
            self.assertEqual(result["validation_render_sha256"], sha256_file(root / "train" / "val_00000.png"))

    def test_inspect_run_rejects_wrong_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.make_run(Path(temporary), "front")
            with self.assertRaisesRegex(ValueError, "validation probe mismatch"):
                inspect_run("left", root)


if __name__ == "__main__":
    unittest.main()
