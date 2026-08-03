import unittest

from radeon_oneloop.phase_targets import Interval, frame_role, normalize_positive_mean


class PhaseTargetTests(unittest.TestCase):
    def test_interval_priority_and_fallback(self):
        intervals = [Interval(10, 20, "recovery"), Interval(21, 30, "correction")]
        self.assertEqual(frame_role(9, intervals, fallback="failed_policy_prefix"), "failed_policy_prefix")
        self.assertEqual(frame_role(10, intervals, fallback="failed_policy_prefix"), "recovery")
        self.assertEqual(frame_role(25, intervals, fallback="failed_policy_prefix"), "correction")

    def test_positive_mean_normalization_preserves_zero(self):
        values = normalize_positive_mean([0.0, 1.0, 3.0])
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual((values[1] + values[2]) / 2.0, 1.0)


if __name__ == "__main__":
    unittest.main()

