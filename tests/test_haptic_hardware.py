import unittest

from sim.genesis_so101.haptic_hardware import (
    FeetechHapticBenchRenderer,
    HapticBenchConfig,
    HapticHardwareError,
)
from sim.genesis_so101.haptic_safety import HapticImpedanceCommand


class FakeBus:
    def __init__(self):
        self.registers = {
            "Torque_Enable": 0,
            "Torque_Limit": 1000,
            "Operating_Mode": 0,
            "Present_Current": 0,
            "Present_Temperature": 30,
            "Present_Voltage": 73,
            "Status": 0,
        }
        self.events = []

    def read(self, register, motor, *, normalize, num_retry=0):
        self.events.append(("read", register, motor, normalize, num_retry))
        return self.registers[register]

    def write(self, register, motor, value, *, normalize=True, num_retry=0):
        self.events.append(("write", register, motor, value, normalize, num_retry))
        self.registers[register] = value

    def enable_torque(self, motors, num_retry=0):
        self.events.append(("enable", tuple(motors), num_retry))
        self.registers["Torque_Enable"] = 1

    def disable_torque(self, motors, num_retry=0):
        self.events.append(("disable", tuple(motors), num_retry))
        self.registers["Torque_Enable"] = 0


def command(*, enabled=True, offset=0.25, torque=30):
    offsets = [0.0] * 12
    limits = [0] * 12
    offsets[4] = offset
    limits[4] = torque
    return HapticImpedanceCommand(enabled, tuple(offsets), tuple(limits), "test")


class HapticHardwareTests(unittest.TestCase):
    def test_bench_is_single_non_gripper_joint_and_low_torque(self):
        with self.assertRaises(ValueError):
            HapticBenchConfig(side="left", motor="gripper")
        with self.assertRaises(ValueError):
            HapticBenchConfig(side="left", motor="wrist_roll", max_torque_limit_raw=31)

    def test_arm_requires_physical_stop_confirmation(self):
        renderer = FeetechHapticBenchRenderer(
            FakeBus(), HapticBenchConfig(side="left", motor="wrist_roll")
        )
        with self.assertRaises(HapticHardwareError):
            renderer.arm((0.0,) * 12, physical_estop_confirmed=False)

    def test_output_is_bounded_and_close_releases_then_restores(self):
        bus = FakeBus()
        renderer = FeetechHapticBenchRenderer(
            bus, HapticBenchConfig(side="left", motor="wrist_roll")
        )
        renderer.arm((0.0,) * 12, physical_estop_confirmed=True)
        renderer.apply(command(torque=999), (0.0,) * 12)
        self.assertEqual(bus.registers["Torque_Limit"], 30)
        renderer.close()
        self.assertEqual(bus.registers["Torque_Enable"], 0)
        self.assertEqual(bus.registers["Torque_Limit"], 1000)
        self.assertTrue(renderer.release_attempted)
        self.assertTrue(renderer.release_verified)
        self.assertTrue(renderer.restore_verified)
        self.assertFalse(renderer.armed)
        disable_index = max(
            index for index, event in enumerate(bus.events) if event[0] == "disable"
        )
        restore_index = max(
            index
            for index, event in enumerate(bus.events)
            if event[:3] == ("write", "Torque_Limit", "wrist_roll")
            and event[3] == 1000
        )
        self.assertLess(disable_index, restore_index)

    def test_health_violation_is_reported(self):
        bus = FakeBus()
        bus.registers["Present_Temperature"] = 50
        renderer = FeetechHapticBenchRenderer(
            bus, HapticBenchConfig(side="right", motor="wrist_flex")
        )
        with self.assertRaises(HapticHardwareError):
            renderer.arm((0.0,) * 12, physical_estop_confirmed=True)

    def test_close_rejects_unverified_torque_release(self):
        bus = FakeBus()
        renderer = FeetechHapticBenchRenderer(
            bus, HapticBenchConfig(side="left", motor="elbow_flex")
        )
        renderer.arm((0.0,) * 12, physical_estop_confirmed=True)
        original_disable = bus.disable_torque

        def ineffective_disable(motors, num_retry=0):
            bus.events.append(("ineffective-disable", tuple(motors), num_retry))

        bus.disable_torque = ineffective_disable
        with self.assertRaisesRegex(HapticHardwareError, "remained enabled"):
            renderer.close()
        self.assertFalse(renderer.release_verified)
        bus.disable_torque = original_disable
        renderer.emergency_release()


if __name__ == "__main__":
    unittest.main()
