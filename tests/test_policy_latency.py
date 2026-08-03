import unittest

from evaluation.policy_latency import latency_summary, percentile


class PolicyLatencyTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)

    def test_summary_preserves_sample_count_and_extrema(self):
        summary = latency_summary([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)
        self.assertEqual(summary["mean_ms"], 2.5)


if __name__ == "__main__":
    unittest.main()
