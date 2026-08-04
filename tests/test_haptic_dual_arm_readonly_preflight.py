import copy
import unittest

from sim.genesis_so101.haptic_arm_readonly_preflight import arm_command_envelope
from sim.genesis_so101.haptic_dual_arm_readonly_preflight import (
    evaluate_dual_hardware_snapshot,
)
from sim.genesis_so101.haptic_hardware import BENCH_MOTORS


def healthy_registers():
    return {
        side: {
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
        for side in ("left", "right")
    }


def envelopes():
    return {
        side: arm_command_envelope(
            side=side,
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
            max_torque_limit_raw=15,
            max_position_offset_deg=0.4,
        )
        for side in ("left", "right")
    }


class HapticDualArmReadonlyPreflightTests(unittest.TestCase):
    def test_accepts_all_ten_healthy_motors(self):
        checks, gates = evaluate_dual_hardware_snapshot(
            action=(0.0,) * 12,
            registers_by_side=healthy_registers(),
            envelopes_by_side=envelopes(),
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(set(gates), {"left", "right"})
        self.assertEqual(set(gates["left"]), set(BENCH_MOTORS))

    def test_one_right_motor_failure_rejects_both_arm_stage(self):
        registers = copy.deepcopy(healthy_registers())
        registers["right"]["shoulder_lift"]["Status"] = 1
        checks, _ = evaluate_dual_hardware_snapshot(
            action=(0.0,) * 12,
            registers_by_side=registers,
            envelopes_by_side=envelopes(),
        )
        self.assertFalse(checks["right_arm_readonly_preflight_accepted"])
        self.assertFalse(checks["right_all_selected_status_clear"])

    def test_one_left_joint_near_limit_rejects_both_arm_stage(self):
        action = [0.0] * 12
        action[2] = 84.7
        checks, gates = evaluate_dual_hardware_snapshot(
            action=tuple(action),
            registers_by_side=healthy_registers(),
            envelopes_by_side=envelopes(),
        )
        self.assertFalse(checks["left_arm_readonly_preflight_accepted"])
        self.assertFalse(
            gates["left"]["elbow_flex"]["bidirectional_model_margin"]
        )


if __name__ == "__main__":
    unittest.main()
