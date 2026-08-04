import unittest

from sim.genesis_so101.haptic_contact_calibration import (
    positive_face_sweep_centres,
    summarize_sweep,
)


class HapticContactCalibrationTest(unittest.TestCase):
    def test_positive_face_centres_encode_clearance_and_penetration(self) -> None:
        centres = positive_face_sweep_centres(
            ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            ((-0.5, -1.0, -1.5), (0.5, 1.0, 1.5)),
            (-2.0, 0.0, 3.0),
        )
        self.assertAlmostEqual(centres[0][0], 1.502)
        self.assertAlmostEqual(centres[1][0], 1.5)
        self.assertAlmostEqual(centres[2][0], 1.497)
        self.assertEqual(centres[0][1:], (1.0, 1.5))

    def test_summary_requires_quiet_baseline_and_two_contact_depths(self) -> None:
        def sample(depth: float, force: float, effort: float):
            return {
                "penetration_mm": depth,
                "object_contact_count_peak": int(force > 0.0),
                "force_n": {
                    "left_median": force,
                    "left_peak": force,
                    "right_median": 0.0,
                    "right_peak": 0.0,
                },
                "joint_effort_abs": {"median": effort, "peak": effort},
                "joint_effort_samples": [[effort] * 6 + [0.0] * 6],
            }

        result = summarize_sweep(
            [sample(-2.0, 0.0, 0.0), sample(0.5, 1.0, 0.8), sample(1.0, 2.0, 1.2)],
            contact_deadband_n=0.5,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["contact_depth_count"], 2)
        self.assertEqual(result["recommended_first_bench_motor"], "shoulder_pan")
        self.assertAlmostEqual(
            result["recommended_simulated_effort_full_scale_p95"], 5.9
        )

    def test_summary_rejects_contact_at_negative_clearance(self) -> None:
        bad = {
            "penetration_mm": -1.0,
            "object_contact_count_peak": 1,
            "force_n": {"left_median": 1.0, "left_peak": 1.0, "right_median": 0.0, "right_peak": 0.0},
            "joint_effort_abs": {"median": 1.0, "peak": 1.0},
            "joint_effort_samples": [[1.0] * 6 + [0.0] * 6],
        }
        contact = dict(bad)
        contact.update({"penetration_mm": 1.0})
        result = summarize_sweep([bad, contact, {**contact, "penetration_mm": 2.0}], contact_deadband_n=0.5)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["negative_clearance_is_quiet"])


if __name__ == "__main__":
    unittest.main()
