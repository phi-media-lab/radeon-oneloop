import copy
import os
import tempfile
import unittest
from pathlib import Path

from radeon_oneloop.train_command import (
    TrainingConfigError,
    absolute_executable_path,
    assert_fair_pair,
    build_command,
)


def config(phase: bool):
    return {
        "experiment": "test",
        "dataset": {"repo_id": "x", "video_backend": "pyav"},
        "policy": {"family": "ACT", "device": "cuda"},
        "optimizer": {"source": "default"},
        "training": {
            "batch_size": 2,
            "steps": 10,
            "num_workers": 0,
            "log_freq": 1,
            "save_freq": 5,
            "output_dir": "ignored",
        },
        "reproducibility": {"seed": 1},
        "method": {
            "phase_aware": phase,
            "intervention": "per_frame_loss_weighting" if phase else None,
        },
    }


class TrainPairTests(unittest.TestCase):
    def test_pair_allows_only_method_difference(self):
        assert_fair_pair(config(False), config(True))

    def test_pair_rejects_budget_difference(self):
        baseline = config(False)
        phase = config(True)
        phase["training"]["steps"] = 11
        with self.assertRaises(TrainingConfigError):
            assert_fair_pair(baseline, phase)

    def test_python_path_keeps_virtualenv_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system_python = root / "system-python"
            system_python.touch()
            venv_python = root / "venv-python"
            venv_python.symlink_to(system_python)

            actual = absolute_executable_path(venv_python)

            self.assertEqual(actual, Path(os.path.abspath(venv_python)))
            self.assertNotEqual(actual, system_python)

    def test_training_command_disables_unrequested_hub_push(self):
        command = build_command(
            config(False),
            python=Path("/venv/bin/python"),
            dataset_root=Path("/dataset"),
            output_dir=Path("/output"),
        )

        self.assertIn("--policy.push_to_hub=false", command)

if __name__ == "__main__":
    unittest.main()
