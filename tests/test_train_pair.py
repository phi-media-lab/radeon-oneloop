import copy
import os
import tempfile
import unittest
from pathlib import Path

from radeon_oneloop.train_command import (
    TrainingConfigError,
    absolute_executable_path,
    assert_fair_pair,
)


def config(phase: bool):
    return {
        "dataset": {"repo_id": "x", "video_backend": "pyav"},
        "policy": {"family": "ACT", "device": "cuda"},
        "optimizer": {"source": "default"},
        "training": {"steps": 10, "output_dir": "ignored"},
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

if __name__ == "__main__":
    unittest.main()
