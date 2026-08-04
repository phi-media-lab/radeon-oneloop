import copy
import unittest

from radeon_oneloop.contracts import ACTION_NAMES
from sim.genesis_so101.haptic_dual_monitor_gate import (
    evaluate_dual_arm_monitor,
)


def evidence():
    authorization = {
        "schema_version": "radeon_oneloop.haptic_stage_authorization.v1",
        "accepted": True,
        "target_stage": "dual_arm_monitor_only",
        "receipt_sha256": "1" * 64,
        "receipt_hash_index_sha256": "2" * 64,
        "receipt_done_sha256": "3" * 64,
        "physical_output_commands": False,
    }
    consumer = {
        "schema_version": "radeon_oneloop.genesis_live_teleop.v1",
        "duration_s": 40.0,
        "ready_file_emitted": True,
        "operator_start_delay_s": 5.0,
        "sim_hz_effective": 119.0,
        "packets": {"accepted": 1200, "rejected": 0},
        "watchdog": {"events": 0, "active_at_end": False},
        "scene_layout": {
            "arrangement": "side_by_side_parallel",
            "base_separation_m": 0.40,
            "left_base_pos_m": [0.20, 0.0, 0.425],
            "right_base_pos_m": [-0.20, 0.0, 0.425],
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
            "packets_sent": 1199,
            "send_errors": 0,
            "physical_output_commands": False,
        },
        "physical_output_commands": False,
    }
    publisher = {
        "schema_version": "radeon_oneloop.leader_publisher.v1",
        "effective_hz": 30.0,
        "samples": 1201,
        "send_errors": 0,
        "action_range": {
            "action_names": list(ACTION_NAMES),
            "samples": 1200,
            "capture_start_gated": True,
            "capture_started": True,
            "minimum": [0.0] * 12,
            "maximum": [4.0, 4.0, 4.0, 4.0, 4.0, 6.0] * 2,
            "span": [4.0, 4.0, 4.0, 4.0, 4.0, 6.0] * 2,
        },
        "haptic_feedback": {
            "mode": "monitor",
            "accepted": 1199,
            "rejected": 0,
            "output_armed_ever": False,
            "output_commands": 0,
            "physical_output_commands": False,
        },
        "physical_output_commands": False,
    }
    return consumer, publisher, authorization


class HapticDualMonitorGateTests(unittest.TestCase):
    def test_accepts_both_exercised_arms_with_zero_output(self):
        report = evaluate_dual_arm_monitor(*evidence())
        self.assertTrue(report["accepted"])
        self.assertTrue(all(report["checks"].values()))

    def test_rejects_unexercised_right_arm(self):
        values = list(copy.deepcopy(evidence()))
        values[1]["action_range"]["span"][6:] = [1.0] * 6
        report = evaluate_dual_arm_monitor(*values)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["right_arm_exercised"])

    def test_rejects_left_right_layout_swap(self):
        values = list(copy.deepcopy(evidence()))
        values[0]["scene_layout"]["left_base_pos_m"] = [-0.20, 0.0, 0.425]
        report = evaluate_dual_arm_monitor(*values)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["parallel_side_by_side_layout"])

    def test_rejects_any_output_arm(self):
        values = list(copy.deepcopy(evidence()))
        values[1]["haptic_feedback"]["output_armed_ever"] = True
        report = evaluate_dual_arm_monitor(*values)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["physical_output_absent"])


if __name__ == "__main__":
    unittest.main()
