import unittest

from sim.genesis_so101.haptic_readonly_preflight import (
    command_envelope,
    evaluate_hardware_snapshot,
    read_register_snapshot,
)


class FakeBus:
    def __init__(self):
        self.values = {
            "Torque_Enable": 0,
            "Operating_Mode": 0,
            "Torque_Limit": 80,
            "Present_Current": -2,
            "Present_Temperature": 31,
            "Present_Voltage": 74,
            "Status": 0,
        }
        self.reads = []

    def read(self, register, motor, *, normalize, num_retry):
        self.reads.append((register, motor, normalize, num_retry))
        return self.values[register]


class HapticReadonlyPreflightTests(unittest.TestCase):
    def test_candidate_envelope_is_single_channel_bounded_and_fail_zero(self):
        result = command_envelope(
            side="left",
            motor="elbow_flex",
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_action_index"], 2)
        self.assertAlmostEqual(result["max_observed_position_offset_deg"], 0.2)
        self.assertEqual(result["max_observed_torque_limit_raw"], 30)
        self.assertTrue(result["checks"]["watchdog_fails_zero"])

    def test_hardware_snapshot_is_read_only_and_requires_torque_disabled(self):
        bus = FakeBus()
        registers = read_register_snapshot(bus, "elbow_flex")
        self.assertEqual(len(bus.reads), 7)
        self.assertTrue(all(read[2] is False for read in bus.reads))
        envelope = command_envelope(
            side="left",
            motor="elbow_flex",
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
        )
        checks = evaluate_hardware_snapshot(
            action=(0.0,) * 12,
            registers=registers,
            envelope=envelope,
        )
        self.assertTrue(all(checks.values()))
        registers["Torque_Enable"] = 1
        checks = evaluate_hardware_snapshot(
            action=(0.0,) * 12,
            registers=registers,
            envelope=envelope,
        )
        self.assertFalse(checks["selected_motor_torque_disabled"])

    def test_selected_joint_requires_margin_for_both_reaction_directions(self):
        envelope = command_envelope(
            side="left",
            motor="elbow_flex",
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
        )
        registers = read_register_snapshot(FakeBus(), "elbow_flex")
        action = [0.0] * 12
        action[2] = 84.0
        checks = evaluate_hardware_snapshot(
            action=action,
            registers=registers,
            envelope=envelope,
        )
        self.assertTrue(checks["selected_position_has_bidirectional_model_margin"])
        action[2] = 84.1
        checks = evaluate_hardware_snapshot(
            action=action,
            registers=registers,
            envelope=envelope,
        )
        self.assertFalse(checks["selected_position_has_bidirectional_model_margin"])


if __name__ == "__main__":
    unittest.main()
