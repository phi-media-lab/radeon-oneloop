import tempfile
import unittest
from pathlib import Path

from gaussian.chunk_artifact import join_artifact, sha256_file, split_artifact


class ChunkArtifactTests(unittest.TestCase):
    def test_split_join_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(bytes(range(251)) * 100)
            chunks = root / "chunks"
            manifest = split_artifact(source, chunks, 1024)
            self.assertGreater(len(manifest["parts"]), 1)
            output = root / "output.bin"
            result = join_artifact(chunks / "manifest.json", output)
            self.assertEqual(result["sha256"], sha256_file(source))
            self.assertEqual(output.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
