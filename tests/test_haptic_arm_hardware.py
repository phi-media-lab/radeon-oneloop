import unittest

from sim.genesis_so101.haptic_arm_hardware import (
    FeetechHapticArmRenderer,
    HapticArmConfig,
)
from sim.genesis_so101.haptic_hardware import BENCH_MOTORS, HapticHardwareError
from sim.genesis_so101.haptic_safety import HapticImpedanceCommand


class FakeBus:
    def __init__(self):
        self.registers = {
            motor: {
                "Torque_Enable": 0,
                "Torque_Limit": 1000,
                "Operating_Mode": 0,
                "Goal_Position": 0.0,
                "Present_Current": 0,
                "Present_Temperature": 30,
                "Present_Voltage": 73,
                "Status": 0,
            }
            for motor in BENCH_MOTORS
        }
        self.events = []

    def read(self, register, motor, *, normalize, num_retry=0):
        self.events.append(("read", register, motor, normalize, num_retry))
        return self.registers[motor][register]

    def write(self, register, motor, value, *, normalize=True, num_retry=0):
        self.events.append(
            ("write", register, motor, value, normalize, num_retry)
        )
        self.registers[motor][register] = value

    def sync_read(self, register, motors, *, normalize=True, num_retry=0):
        self.events.append(
            ("sync_read", register, tuple(motors), normalize, num_retry)
        )
        return {motor: self.registers[motor][register] for motor in motors}

    def sync_write(self, register, values, *, normalize=True, num_retry=0):
        self.events.append(
            ("sync_write", register, dict(values), normalize, num_retry)
        )
        for motor, value in values.items():
            self.registers[motor][register] = value

    def enable_torque(self, motors, num_retry=0):
        self.events.append(("enable", tuple(motors), num_retry))
        for motor in motors:
            self.registers[motor]["Torque_Enable"] = 1

    def disable_torque(self, motors, num_retry=0):
        self.events.append(("disable", tuple(motors), num_retry))
        for motor in motors:
            self.registers[motor]["Torque_Enable"] = 0


def command(*, torque=20, offset=0.1):
    offsets = [0.0] * 12
    limits = [0] * 12
    for index in range(5):
        offsets[index] = offset
        limits[index] = torque
    return HapticImpedanceCommand(
        enabled=True,
        position_offset_deg=tuple(offsets),
        torque_limit_raw=tuple(limits),
        reason="test",
    )


class HapticArmHardwareTests(unittest.TestCase):
    def test_config_is_five_joint_low_torque_only(self):
        self.assertEqual(HapticArmConfig(side="left").action_indices, (0, 1, 2, 3, 4))
        self.assertEqual(HapticArmConfig(side="right").action_indices, (6, 7, 8, 9, 10))
        with self.assertRaises(ValueError):
            HapticArmConfig(side="left", max_torque_limit_raw=21)

    def test_arm_requires_reachable_physical_stop(self):
        renderer = FeetechHapticArmRenderer(FakeBus(), HapticArmConfig(side="left"))
        with self.assertRaises(HapticHardwareError):
            renderer.arm((0.0,) * 12, physical_estop_confirmed=False)

    def test_close_before_arm_is_strictly_read_only(self):
        bus = FakeBus()
        renderer = FeetechHapticArmRenderer(bus, HapticArmConfig(side="left"))
        renderer.close()
        self.assertEqual(bus.events, [])
        self.assertFalse(renderer.release_attempted)

    def test_batch_output_is_bounded_then_close_reliably_releases(self):
        bus = FakeBus()
        renderer = FeetechHapticArmRenderer(bus, HapticArmConfig(side="left"))
        renderer.arm((0.0,) * 12, physical_estop_confirmed=True)
        renderer.apply(command(torque=999), (0.0,) * 12)
        self.assertTrue(
            all(
                bus.registers[motor]["Torque_Limit"] == 20
                for motor in BENCH_MOTORS
            )
        )
        renderer.close()
        self.assertTrue(
            all(
                bus.registers[motor]["Torque_Enable"] == 0
                for motor in BENCH_MOTORS
            )
        )
        self.assertTrue(
            all(
                bus.registers[motor]["Torque_Limit"] == 1000
                for motor in BENCH_MOTORS
            )
        )
        self.assertTrue(renderer.release_verified)
        self.assertTrue(renderer.restore_verified)
        self.assertFalse(renderer.armed)

    def test_any_motor_health_failure_rejects_arm(self):
        bus = FakeBus()
        bus.registers["wrist_flex"]["Present_Temperature"] = 50
        renderer = FeetechHapticArmRenderer(bus, HapticArmConfig(side="left"))
        with self.assertRaisesRegex(HapticHardwareError, "wrist_flex temperature"):
            renderer.arm((0.0,) * 12, physical_estop_confirmed=True)

    def test_close_rejects_unverified_release(self):
        bus = FakeBus()
        renderer = FeetechHapticArmRenderer(bus, HapticArmConfig(side="left"))
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
