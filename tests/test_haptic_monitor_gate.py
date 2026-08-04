import unittest

from radeon_oneloop.contracts import ACTION_NAMES
from sim.genesis_so101.haptic_monitor_gate import evaluate_single_arm_monitor
from sim.genesis_so101.leader_publisher import ActionRangeTracker


def authorization():
    return {
        "schema_version": "radeon_oneloop.haptic_stage_authorization.v1",
        "accepted": True,
        "target_stage": "single_arm_monitor_only",
        "receipt_sha256": "1" * 64,
        "receipt_hash_index_sha256": "2" * 64,
        "receipt_done_sha256": "3" * 64,
        "physical_output_commands": False,
    }


def consumer():
    return {
        "schema_version": "radeon_oneloop.genesis_live_teleop.v1",
        "duration_s": 30.0,
        "ready_file_emitted": True,
        "operator_start_delay_s": 5.0,
        "sim_hz_effective": 119.0,
        "packets": {"accepted": 900, "rejected": 0},
        "watchdog": {"events": 0, "active_at_end": False},
        "scene_layout": {
            "arrangement": "side_by_side_parallel",
            "base_separation_m": 0.40,
            "shared_base_euler_deg": [0.0, 0.0, 0.0],
        },
        "input_clamping": {
            "processed_packets_with_clamping": 0,
            "processed_values_clamped": 0,
            "max_abs_delta": 0.0,
        },
        "tracking_error": {"max_abs": 0.25},
        "haptic_feedback": {
            "enabled": True,
            "packets_sent": 899,
            "send_errors": 0,
            "physical_output_commands": False,
        },
        "physical_output_commands": False,
    }


def publisher():
    return {
        "schema_version": "radeon_oneloop.leader_publisher.v1",
        "effective_hz": 30.0,
        "samples": 901,
        "send_errors": 0,
        "action_range": {
            "action_names": list(ACTION_NAMES),
            "samples": 900,
            "capture_start_gated": True,
            "capture_started": True,
            "minimum": [0.0] * 12,
            "maximum": [4.0, 4.0, 4.0, 4.0, 4.0, 6.0] + [1.0] * 6,
            "span": [4.0, 4.0, 4.0, 4.0, 4.0, 6.0] + [1.0] * 6,
        },
        "haptic_feedback": {
            "mode": "monitor",
            "accepted": 899,
            "rejected": 0,
            "output_armed_ever": False,
            "output_commands": 0,
            "physical_output_commands": False,
        },
        "physical_output_commands": False,
    }


class ActionRangeTrackerTests(unittest.TestCase):
    def test_tracks_per_channel_minimum_maximum_and_span(self):
        tracker = ActionRangeTracker()
        tracker.update(tuple(float(index) for index in range(12)))
        tracker.update(tuple(float(index - 2) for index in range(12)))
        report = tracker.as_dict()
        self.assertEqual(report["action_names"], list(ACTION_NAMES))
        self.assertEqual(report["span"], [2.0] * 12)


class HapticMonitorGateTests(unittest.TestCase):
    def test_accepts_exercised_left_arm_with_quiet_right_arm(self):
        report = evaluate_single_arm_monitor(
            consumer(), publisher(), authorization(), expected_side="left"
        )
        self.assertTrue(report["accepted"])
        self.assertFalse(report["physical_output_commands"])
        self.assertEqual(report["next_candidate_stage"], "single_arm_physical")

    def test_does_not_mistake_quiet_right_arm_for_exercised_arm(self):
        report = evaluate_single_arm_monitor(
            consumer(), publisher(), authorization(), expected_side="right"
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["selected_arm_exercised"])

    def test_rejects_any_physical_output_claim(self):
        observed = publisher()
        observed["physical_output_commands"] = True
        report = evaluate_single_arm_monitor(
            consumer(), observed, authorization(), expected_side="left"
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["physical_output_absent"])

    def test_rejects_input_clamping(self):
        observed = consumer()
        observed["input_clamping"]["processed_packets_with_clamping"] = 1
        report = evaluate_single_arm_monitor(
            observed, publisher(), authorization(), expected_side="left"
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["zero_input_clamping"])

    def test_rejects_wrong_stage_authorization(self):
        auth = authorization()
        auth["target_stage"] = "single_arm_physical"
        report = evaluate_single_arm_monitor(
            consumer(), publisher(), auth, expected_side="left"
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["authorization_schema_and_target"])


if __name__ == "__main__":
    unittest.main()
