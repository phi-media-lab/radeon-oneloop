import unittest

from sim.genesis_so101.haptic_bench_gate import evaluate_bench


def evidence():
    publisher = {
        "schema_version": "radeon_oneloop.leader_publisher.v1",
        "effective_hz": 29.9,
        "send_errors": 0,
        "physical_output_commands": True,
        "haptic_feedback": {
            "mode": "bench-single-joint",
            "accepted": 300,
            "rejected": 0,
            "bench_selection": {
                "side": "left",
                "motor": "elbow_flex",
                "simulated_effort_full_scale": 0.6727447137236594,
            },
            "output_armed_ever": True,
            "output_commands": 290,
            "latest_health": {
                "present_current_raw": 2,
                "present_temperature_c": 31,
                "present_voltage_raw": 73,
                "status": 0,
            },
            "peak_abs_current_raw": 2,
            "peak_temperature_c": 31,
            "shutdown_error": None,
            "release_attempted": True,
            "release_verified": True,
            "restore_verified": True,
            "output_armed_at_shutdown": False,
        },
    }
    sender = {
        "schema_version": "radeon_oneloop.haptic_bench_sender.v1",
        "side": "left",
        "motor": "elbow_flex",
        "duration_s": 10.5,
        "packets_sent": 315,
        "effective_hz": 30.0,
        "reaction_effort": 0.1345489427447319,
        "physical_output_commands": False,
    }
    return publisher, sender


class HapticBenchGateTests(unittest.TestCase):
    def evaluate(self, publisher, sender):
        return evaluate_bench(
            publisher,
            sender,
            expected_side="left",
            expected_motor="elbow_flex",
            expected_full_scale=0.6727447137236594,
            expected_reaction_effort=0.1345489427447319,
        )

    def test_accepts_complete_safe_evidence(self):
        result = self.evaluate(*evidence())
        self.assertTrue(result["accepted"])
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_unverified_release(self):
        publisher, sender = evidence()
        publisher["haptic_feedback"]["release_verified"] = False
        result = self.evaluate(publisher, sender)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["verified_fail_zero_shutdown"])

    def test_rejects_wrong_calibration_scale(self):
        publisher, sender = evidence()
        publisher["haptic_feedback"]["bench_selection"][
            "simulated_effort_full_scale"
        ] = 3.35
        result = self.evaluate(publisher, sender)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["selection_matches"])


if __name__ == "__main__":
    unittest.main()
