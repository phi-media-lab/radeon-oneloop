import unittest

from gaussian.extract_object_refs import parse_episode_indices, sample_timestamps


class ExtractObjectRefsTests(unittest.TestCase):
    def test_samples_are_even_and_avoid_endpoints(self):
        self.assertEqual(sample_timestamps(10.0, 20.0, 4), (12.0, 14.0, 16.0, 18.0))

    def test_episode_csv_is_unique_and_nonnegative(self):
        self.assertEqual(parse_episode_indices("17, 20,22"), (17, 20, 22))
        for invalid in ("", "-1,2", "2,2"):
            with self.assertRaises(ValueError):
                parse_episode_indices(invalid)


if __name__ == "__main__":
    unittest.main()
