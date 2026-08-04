import argparse
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian.completion_candidate import build_candidate, validate_candidate


def write_point_ply(path: Path, count: int, *, with_support: bool = False) -> None:
    properties = [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
    ]
    if with_support:
        properties.extend(
            [
                ("uchar", "cross_view_source_count"),
                ("uchar", "silhouette_support_count"),
            ]
        )
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header.extend(f"property {kind} {name}" for kind, name in properties)
    header.append("end_header")
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        for index in range(count):
            handle.write(struct.pack("<fff", index * 0.001, 0.0, 0.0))
            if with_support:
                handle.write(struct.pack("BB", 1 + index % 2, 2))


class CompletionCandidateTests(unittest.TestCase):
    def build_args(self, root: Path, **overrides):
        values = {
            "observed_ply": root / "observed.ply",
            "completed_ply": root / "completed.ply",
            "output": root / "candidate",
            "point_set_role": "generated_fill_only",
            "confidence_npy": None,
            "source_labels_npy": None,
            "default_generated_confidence": None,
            "conditioning_json": None,
            "canonical_transform_json": None,
            "checkpoint": None,
            "generator_name": "test_completion",
            "generator_version": "1",
            "host_role": "phi-amd-work",
            "accelerator": "AMD Instinct MI300X",
            "seed": 7,
            "object_height_m": 0.095,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_builds_and_validates_evidence_derived_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_point_ply(root / "observed.ply", 3)
            write_point_ply(root / "completed.ply", 4, with_support=True)
            result = build_candidate(self.build_args(root))
            self.assertTrue(result["valid"])
            self.assertEqual(result["completion_vertices"], 4)
            self.assertEqual(result["source_label_counts"]["generated"], 4)
            confidence = np.load(root / "candidate" / "confidence.npy", allow_pickle=False)
            np.testing.assert_allclose(confidence, [0.375, 0.5, 0.375, 0.5])
            manifest = json.loads((root / "candidate" / "manifest.json").read_text())
            self.assertFalse(manifest["formal"])
            self.assertFalse(manifest["eligible_for_heldout_real_metrics"])

    def test_rejects_full_candidate_without_explicit_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_point_ply(root / "observed.ply", 3)
            write_point_ply(root / "completed.ply", 4, with_support=True)
            with self.assertRaisesRegex(ValueError, "requires --source-labels-npy"):
                build_candidate(self.build_args(root, point_set_role="full_candidate"))

    def test_hash_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_point_ply(root / "observed.ply", 3)
            write_point_ply(root / "completed.ply", 4, with_support=True)
            build_candidate(self.build_args(root))
            with (root / "candidate" / "conditioning.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "size or hash mismatch"):
                validate_candidate(root / "candidate")

    def test_rejects_private_absolute_path_in_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_point_ply(root / "observed.ply", 3)
            write_point_ply(root / "completed.ply", 4, with_support=True)
            conditioning = root / "conditioning.json"
            conditioning.write_text(json.dumps({"input": "/private/raw/image.png"}))
            with self.assertRaisesRegex(ValueError, "must not embed"):
                build_candidate(self.build_args(root, conditioning_json=conditioning))


if __name__ == "__main__":
    unittest.main()
