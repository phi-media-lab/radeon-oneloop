import json
import tempfile
import unittest
from pathlib import Path

from gaussian.vksplat_train import (
    FROZEN_MEANS_LR,
    freeze_geometry_learning_rates,
    load_trainer,
    validate_dataset,
)


class GaussianContractTests(unittest.TestCase):
    def test_colmap_dataset_fingerprint_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "sparse/0").mkdir(parents=True)
            for index in range(8):
                (root / "images" / f"{index:03d}.png").write_bytes(bytes([index]))
            for name in ("cameras", "images", "points3D"):
                (root / "sparse/0" / f"{name}.bin").write_bytes(name.encode())
            first = validate_dataset(root, "images", "sparse/0")
            second = validate_dataset(root, "images", "sparse/0")
            self.assertEqual(first, second)
            self.assertEqual(first["images"], 8)
            self.assertEqual(first["model_format"], "bin")

    def test_capture_schema_is_valid_json(self):
        schema = json.loads(Path("gaussian/workspace_capture.schema.json").read_text())
        self.assertIn("scale_anchor", schema["required"])

    def test_four_view_object_dataset_requires_matching_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            (root / "sparse/0").mkdir(parents=True)
            for index in range(4):
                (root / "images" / f"anchor_{index}.png").write_bytes(bytes([index]))
                (root / "masks" / f"anchor_{index}.png").write_bytes(bytes([index + 1]))
            for name in ("cameras", "images", "points3D"):
                (root / "sparse/0" / f"{name}.txt").write_text(name)
            value = validate_dataset(
                root,
                "images",
                "sparse/0",
                min_images=4,
                mask_dir="masks",
            )
            self.assertEqual(value["images"], 4)
            self.assertEqual(value["masks"], 4)

    def test_pinned_trainer_shader_path_keeps_upstream_checkout_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            trainer_path = source / "vksplat" / "simple_trainer.py"
            trainer_path.parent.mkdir(parents=True)
            trainer_path.write_text(
                "import os\n"
                "import random\n"
                "import numpy as np\n"
                "def join_dir(parent, child):\n"
                "    value = os.path.join(parent, child)\n"
                "    return value if value.endswith(os.sep) else value + os.sep\n",
                encoding="utf-8",
            )
            trainer = load_trainer(source)
            self.assertTrue(trainer.join_dir("/tmp", "shader").endswith("shader//"))
            self.assertTrue(trainer.join_dir("/tmp", "output").endswith("output/"))
            self.assertFalse(trainer.join_dir("/tmp", "output").endswith("output//"))

    def test_formal_object_runner_does_not_commit_private_dataset_path(self):
        runner = Path("ops/run_formal_object_vksplat_train.sh").read_text()
        self.assertIn("ONELOOP_OBJECT_DATASET:?", runner)
        self.assertNotIn("/root/radeon-oneloop-data", runner)
        self.assertIn("--formal", runner)

    def test_geometry_freeze_zeroes_only_center_and_shape_rates(self):
        class Config:
            means_lr = 1.6e-4
            means_lr_final = 1.6e-6
            scales_lr = 5.0e-3
            quats_lr = 1.0e-3
            features_dc_lr = 2.5e-3
            opacities_lr = 5.0e-2

        config = Config()
        original = freeze_geometry_learning_rates(config)
        self.assertEqual(
            original,
            {
                "means_lr": 1.6e-4,
                "means_lr_final": 1.6e-6,
                "scales_lr": 5.0e-3,
                "quats_lr": 1.0e-3,
            },
        )
        self.assertEqual(config.means_lr, FROZEN_MEANS_LR)
        self.assertEqual(config.means_lr_final, FROZEN_MEANS_LR)
        self.assertEqual(config.scales_lr, 0.0)
        self.assertEqual(config.quats_lr, 0.0)
        self.assertEqual(config.features_dc_lr, 2.5e-3)
        self.assertEqual(config.opacities_lr, 5.0e-2)
