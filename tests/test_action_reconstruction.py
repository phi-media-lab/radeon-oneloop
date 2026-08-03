import unittest

from evaluation.action_reconstruction import evenly_spaced_indices, scalar_summary


class ActionReconstructionTests(unittest.TestCase):
    def test_even_spacing_is_deterministic_and_keeps_endpoints(self):
        self.assertEqual(evenly_spaced_indices(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices([9, 3, 6, 0], 4), [0, 3, 6, 9])

    def test_limit_larger_than_population_keeps_every_value(self):
        self.assertEqual(evenly_spaced_indices([3, 1, 2], 8), [1, 2, 3])

    def test_scalar_summary_has_unitless_names(self):
        summary = scalar_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertNotIn("mean_ms", summary)


if __name__ == "__main__":
    unittest.main()
