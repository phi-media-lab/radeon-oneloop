import unittest

from sim.genesis_so101.haptic_intervention import (
    StableSafePoseConfig,
    StableSafePoseGate,
)
from sim.genesis_so101.haptic_hardware import BENCH_MOTORS
from sim.genesis_so101.leader_publisher import _build_same_process_preflight


class HealthyBus:
    values = {
        "Torque_Enable": 0,
        "Operating_Mode": 0,
        "Torque_Limit": 1000,
        "Present_Current": 0,
        "Present_Temperature": 32,
        "Present_Voltage": 73,
        "Status": 0,
    }

    def read(self, register, motor, **kwargs):
        del motor, kwargs
        return self.values[register]


class StableSafePoseGateTests(unittest.TestCase):
    def setUp(self):
        self.config = StableSafePoseConfig(
            side="left", hold_s=0.4, max_span_deg=2.0
        )
        self.gate = StableSafePoseGate(self.config)

    def test_requires_continuous_stable_safe_hold(self):
        action = [0.0] * 12
        now_ns = 1_000_000_000
        for _ in range(12):
            self.assertFalse(self.gate.update(action, now_ns=now_ns))
            now_ns += 34_000_000
        self.assertTrue(self.gate.update(action, now_ns=now_ns))
        report = self.gate.as_dict(now_ns=now_ns)
        self.assertTrue(report["candidate_ready"])
        self.assertGreaterEqual(report["stable_duration_s"], 0.4)
        self.assertTrue(report["serial_connection_preserved_for_arm"])

    def test_out_of_margin_elbow_resets_hold(self):
        action = [0.0] * 12
        action[2] = 96.0
        self.assertFalse(self.gate.update(action, now_ns=1_000_000_000))
        report = self.gate.as_dict(now_ns=1_000_000_000)
        self.assertFalse(report["candidate_ready"])
        self.assertFalse(report["latest_position_checks"]["elbow_flex"])
        self.assertEqual(report["last_reset_reason"], "outside_bidirectional_margin")

    def test_motion_larger_than_span_restarts_hold(self):
        action = [0.0] * 12
        self.assertFalse(self.gate.update(action, now_ns=1_000_000_000))
        action[1] = 2.1
        self.assertFalse(self.gate.update(action, now_ns=1_200_000_000))
        report = self.gate.as_dict(now_ns=1_200_000_000)
        self.assertEqual(report["reset_count"], 1)
        self.assertEqual(report["last_reset_reason"], "motion_span_exceeded")
        self.assertEqual(report["stable_duration_s"], 0.0)

    def test_right_side_uses_right_action_indices(self):
        gate = StableSafePoseGate(StableSafePoseConfig(side="right"))
        action = [0.0] * 12
        action[8] = 96.0
        self.assertFalse(gate.update(action, now_ns=1_000_000_000))
        report = gate.as_dict(now_ns=1_000_000_000)
        self.assertFalse(report["latest_position_checks"]["elbow_flex"])

    def test_single_joint_mode_ignores_unselected_arm_pose(self):
        gate = StableSafePoseGate(
            StableSafePoseConfig(side="left", motors=("elbow_flex",))
        )
        action = [0.0] * 12
        action[1] = -103.0
        now_ns = 1_000_000_000
        for _ in range(13):
            ready = gate.update(action, now_ns=now_ns)
            now_ns += 34_000_000
        self.assertTrue(ready)
        report = gate.as_dict(now_ns=now_ns - 34_000_000)
        self.assertEqual(report["motors"], ["elbow_flex"])

    def test_builds_single_joint_preflight_at_same_process_boundary(self):
        gate = StableSafePoseGate(
            StableSafePoseConfig(
                side="left",
                motors=("elbow_flex",),
                hold_s=0.1,
            )
        )
        action = (0.0,) * 12
        gate.update(action, now_ns=1_000_000_000)
        self.assertTrue(gate.update(action, now_ns=1_100_000_000))
        report = _build_same_process_preflight(
            output_mode="bench-single-joint",
            side="left",
            motor="elbow_flex",
            selected_bus=HealthyBus(),
            action=action,
            intervention_gate=gate,
            now_ns=1_100_000_000,
            elapsed_s=0.1,
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
            max_torque_limit_raw=30,
            max_position_offset_deg=1.0,
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(
            report["schema_version"],
            "radeon_oneloop.haptic_readonly_preflight.v1",
        )
        self.assertEqual(report["selected_register_reads"], 7)
        self.assertEqual(report["intervention"]["motors"], ["elbow_flex"])

    def test_builds_five_joint_preflight_at_same_process_boundary(self):
        gate = StableSafePoseGate(
            StableSafePoseConfig(side="left", hold_s=0.1)
        )
        action = (0.0,) * 12
        gate.update(action, now_ns=1_000_000_000)
        self.assertTrue(gate.update(action, now_ns=1_100_000_000))
        report = _build_same_process_preflight(
            output_mode="physical-single-arm",
            side="left",
            motor=None,
            selected_bus=HealthyBus(),
            action=action,
            intervention_gate=gate,
            now_ns=1_100_000_000,
            elapsed_s=0.1,
            simulated_effort_full_scale=0.6727447137236594,
            reaction_effort=0.1345489427447319,
            max_torque_limit_raw=20,
            max_position_offset_deg=0.5,
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(
            report["schema_version"],
            "radeon_oneloop.haptic_arm_readonly_preflight.v1",
        )
        self.assertEqual(report["selected_register_reads"], 35)
        self.assertEqual(report["intervention"]["motors"], list(BENCH_MOTORS))


if __name__ == "__main__":
    unittest.main()
