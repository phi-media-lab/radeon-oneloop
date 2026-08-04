import unittest

from sim.genesis_so101.haptic_safety import SafeHapticController
from sim.genesis_so101.live_protocol import HapticFeedbackPacket


def feedback(*, effort=1.0, force=2.0) -> HapticFeedbackPacket:
    return HapticFeedbackPacket(
        sequence_id=0,
        captured_monotonic_ns=1,
        captured_unix_ns=1,
        joint_reaction_effort=(effort,) * 12,
        contact_force_n=(force, force),
    )


class HapticSafetyTests(unittest.TestCase):
    def test_output_is_disabled_until_physical_estop_is_confirmed(self) -> None:
        controller = SafeHapticController()
        command = controller.update(feedback(), arrival_monotonic_ns=1)
        self.assertFalse(command.enabled)
        self.assertEqual(command.position_offset_deg, (0.0,) * 12)
        with self.assertRaises(RuntimeError):
            controller.arm(physical_estop_confirmed=False)

    def test_contact_command_is_bounded_and_slew_limited(self) -> None:
        controller = SafeHapticController()
        controller.arm(physical_estop_confirmed=True)
        command = controller.update(feedback(effort=100.0), arrival_monotonic_ns=1)
        self.assertTrue(command.enabled)
        self.assertAlmostEqual(max(command.position_offset_deg), 0.075)
        self.assertEqual(max(command.torque_limit_raw), 80)

    def test_no_contact_produces_zero_impedance(self) -> None:
        controller = SafeHapticController()
        controller.arm(physical_estop_confirmed=True)
        command = controller.update(feedback(force=0.1), arrival_monotonic_ns=1)
        self.assertEqual(command.position_offset_deg, (0.0,) * 12)
        self.assertEqual(command.torque_limit_raw, (0,) * 12)

    def test_watchdog_disarms_and_fails_zero(self) -> None:
        controller = SafeHapticController()
        controller.arm(physical_estop_confirmed=True)
        controller.update(feedback(), arrival_monotonic_ns=1_000_000_000)
        command = controller.watchdog(now_monotonic_ns=1_101_000_000)
        self.assertIsNotNone(command)
        assert command is not None
        self.assertFalse(command.enabled)
        self.assertEqual(command.reason, "feedback_timeout")

    def test_estop_latches_until_new_controller_is_created(self) -> None:
        controller = SafeHapticController()
        controller.arm(physical_estop_confirmed=True)
        controller.latch_estop()
        with self.assertRaises(RuntimeError):
            controller.arm(physical_estop_confirmed=True)


if __name__ == "__main__":
    unittest.main()
