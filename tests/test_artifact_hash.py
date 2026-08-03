from pathlib import Path
import tempfile
import unittest

from radeon_oneloop.artifact_hash import tree_sha256


class ArtifactHashTests(unittest.TestCase):
    def test_hash_is_path_ordered_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "z.txt").write_text("last", encoding="utf-8")
            (root / "nested" / "a.txt").write_text("first", encoding="utf-8")

            first, records = tree_sha256(root)
            second, _ = tree_sha256(root)
            self.assertEqual(first, second)
            self.assertEqual(
                [record["path"] for record in records],
                ["nested/a.txt", "z.txt"],
            )

            (root / "z.txt").write_text("changed", encoding="utf-8")
            changed, _ = tree_sha256(root)
            self.assertNotEqual(first, changed)

    def test_empty_tree_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "empty"):
                tree_sha256(Path(directory))


if __name__ == "__main__":
    unittest.main()
