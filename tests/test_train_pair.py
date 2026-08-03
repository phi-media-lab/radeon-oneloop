import copy
import unittest

from radeon_oneloop.train_command import TrainingConfigError, assert_fair_pair


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


if __name__ == "__main__":
    unittest.main()
