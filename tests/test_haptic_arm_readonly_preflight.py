import unittest

from sim.genesis_so101.haptic_arm_readonly_preflight import (
    arm_command_envelope,
    evaluate_arm_hardware_snapshot,
)
from sim.genesis_so101.haptic_hardware import BENCH_MOTORS


def healthy_registers():
    return {
        motor: {
            "Torque_Enable": 0,
            "Operating_Mode": 0,
            "Torque_Limit": 1000,
            "Present_Current": 0,
            "Present_Temperature": 32,
            "Present_Voltage": 73,
            "Status": 0,
        }
        for motor in BENCH_MOTORS
    }


class HapticArmReadonlyPreflightTests(unittest.TestCase):
    def test_arm_envelope_selects_five_joints_and_fails_zero(self):
        envelope = arm_command_envelope(
            side="left",
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
        )
        self.assertTrue(envelope["accepted"])
        self.assertEqual(envelope["selected_action_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(envelope["max_torque_limit_raw"], 20)
        self.assertTrue(envelope["checks"]["watchdog_fails_zero"])
        self.assertTrue(envelope["checks"]["unselected_channels_zero"])

    def test_all_five_joints_must_be_healthy_and_inside_margin(self):
        envelope = arm_command_envelope(
            side="left",
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
        )
        checks, per_motor = evaluate_arm_hardware_snapshot(
            action=(0.0,) * 12,
            side="left",
            registers_by_motor=healthy_registers(),
            envelope=envelope,
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(set(per_motor), set(BENCH_MOTORS))

        registers = healthy_registers()
        registers["wrist_flex"]["Torque_Enable"] = 1
        checks, _ = evaluate_arm_hardware_snapshot(
            action=(0.0,) * 12,
            side="left",
            registers_by_motor=registers,
            envelope=envelope,
        )
        self.assertFalse(checks["all_selected_motors_torque_disabled"])

    def test_any_joint_near_limit_rejects_entire_arm(self):
        envelope = arm_command_envelope(
            side="right",
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
        )
        action = [0.0] * 12
        action[8] = 84.6
        checks, per_motor = evaluate_arm_hardware_snapshot(
            action=action,
            side="right",
            registers_by_motor=healthy_registers(),
            envelope=envelope,
        )
        self.assertFalse(
            checks["all_selected_positions_have_bidirectional_model_margin"]
        )
        self.assertFalse(per_motor["elbow_flex"]["bidirectional_model_margin"])


if __name__ == "__main__":
    unittest.main()
