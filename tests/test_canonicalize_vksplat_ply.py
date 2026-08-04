import unittest
from argparse import Namespace
import json
from pathlib import Path
import tempfile

import numpy as np

from gaussian.canonicalize_vksplat_ply import canonicalize, inverse_similarity


class CanonicalizeVkSplatPlyTests(unittest.TestCase):
    def test_inverse_similarity_recovers_scale_rotation_and_translation(self):
        angle = np.pi / 2.0
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        original_to_normalized = np.eye(4)
        original_to_normalized[:3, :3] = 2.0 * rotation
        original_to_normalized[:3, 3] = [1.0, 2.0, 3.0]
        scale, inverse_rotation, translation = inverse_similarity(original_to_normalized)
        self.assertAlmostEqual(scale, 0.5)
        np.testing.assert_allclose(inverse_rotation, rotation.T, atol=1.0e-7)
        expected = np.linalg.inv(original_to_normalized)
        np.testing.assert_allclose(translation, expected[:3, 3], atol=1.0e-7)

    def test_lineage_arguments_are_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train.json"
            train.write_text(json.dumps({"dataparser_transform": np.eye(4).tolist()}))
            training_manifest = root / "training.json"
            training_manifest.write_text("{}")
            args = Namespace(
                ply=root / "missing.ply",
                train_json=train,
                output=root / "output.ply",
                output_provenance=root / "provenance.json",
                training_run_manifest=training_manifest,
                training_metrics=None,
                training_config=None,
                vksplat_commit=None,
                dataset_manifest=None,
                formal=False,
                host_role="unspecified_nonformal",
            )
            with self.assertRaisesRegex(ValueError, "supplied together"):
                canonicalize(args)

    def test_formal_canonicalization_requires_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train.json"
            train.write_text(json.dumps({"dataparser_transform": np.eye(4).tolist()}))
            args = Namespace(
                ply=root / "missing.ply",
                train_json=train,
                output=root / "output.ply",
                output_provenance=root / "provenance.json",
                training_run_manifest=None,
                training_metrics=None,
                training_config=None,
                vksplat_commit=None,
                dataset_manifest=None,
                formal=True,
                host_role="radeon_c_gpu0_gfx1100_formal",
            )
            with self.assertRaisesRegex(ValueError, "complete training lineage"):
                canonicalize(args)


if __name__ == "__main__":
    unittest.main()
