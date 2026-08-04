"""Fail-safe single-joint STS3215 renderer for the first haptic bench gate.

This module deliberately contains no LeRobot imports.  It accepts an already
connected Feetech bus and is usable only for one non-gripper joint, at a low
torque limit, for a time-bounded bench run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .haptic_safety import HapticImpedanceCommand


ARM_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
BENCH_MOTORS = ARM_MOTORS[:-1]


class HapticHardwareError(RuntimeError):
    """Raised after the renderer has attempted a fail-safe motor release."""


@dataclass(frozen=True)
class HapticBenchConfig:
    side: str
    motor: str
    max_torque_limit_raw: int = 30
    max_abs_current_raw: int = 150
    max_temperature_c: int = 45
    min_voltage_raw: int = 60
    max_voltage_raw: int = 84

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if self.motor not in BENCH_MOTORS:
            raise ValueError(
                "the first haptic bench permits one arm joint; gripper is excluded"
            )
        if not 1 <= self.max_torque_limit_raw <= 30:
            raise ValueError("first-bench torque limit must be between 1 and 30")
        if self.max_abs_current_raw <= 0:
            raise ValueError("max_abs_current_raw must be positive")
        if self.max_temperature_c <= 0:
            raise ValueError("max_temperature_c must be positive")
        if not 0 < self.min_voltage_raw < self.max_voltage_raw:
            raise ValueError("invalid voltage bounds")

    @property
    def action_index(self) -> int:
        side_offset = 0 if self.side == "left" else len(ARM_MOTORS)
        return side_offset + ARM_MOTORS.index(self.motor)


class FeetechHapticBenchRenderer:
    """Render one bounded impedance channel and release torque on every exit."""

    def __init__(self, bus: Any, config: HapticBenchConfig) -> None:
        self.bus = bus
        self.config = config
        self._armed = False
        self._original_torque_limit: int | None = None
        self.output_commands = 0
        self.peak_abs_current_raw = 0
        self.peak_temperature_c = 0
        self.release_attempted = False
        self.release_verified = False
        self.restore_verified = False

    @property
    def armed(self) -> bool:
        return self._armed

    def _read_raw(self, register: str) -> int:
        return int(
            self.bus.read(
                register,
                self.config.motor,
                normalize=False,
                num_retry=2,
            )
        )

    def _write(self, register: str, value: int | float, *, normalize: bool) -> None:
        self.bus.write(
            register,
            self.config.motor,
            value,
            normalize=normalize,
            num_retry=2,
        )

    def check_health(self) -> dict[str, int]:
        current = self._read_raw("Present_Current")
        temperature = self._read_raw("Present_Temperature")
        voltage = self._read_raw("Present_Voltage")
        status = self._read_raw("Status")
        self.peak_abs_current_raw = max(self.peak_abs_current_raw, abs(current))
        self.peak_temperature_c = max(self.peak_temperature_c, temperature)
        if abs(current) > self.config.max_abs_current_raw:
            raise HapticHardwareError(f"current limit exceeded: {current}")
        if temperature > self.config.max_temperature_c:
            raise HapticHardwareError(f"temperature limit exceeded: {temperature} C")
        if not self.config.min_voltage_raw <= voltage <= self.config.max_voltage_raw:
            raise HapticHardwareError(f"voltage outside bounds: {voltage}")
        if status != 0:
            raise HapticHardwareError(f"motor status is nonzero: {status}")
        return {
            "present_current_raw": current,
            "present_temperature_c": temperature,
            "present_voltage_raw": voltage,
            "status": status,
        }

    def arm(self, current_action: Sequence[float], *, physical_estop_confirmed: bool) -> None:
        if self._armed:
            raise HapticHardwareError("renderer is already armed")
        if not physical_estop_confirmed:
            raise HapticHardwareError(
                "a reachable physical power cut/emergency stop must be confirmed"
            )
        if len(current_action) != 12:
            raise HapticHardwareError("current action must contain 12 values")
        if self._read_raw("Torque_Enable") != 0:
            raise HapticHardwareError("selected motor torque must be disabled before arming")
        if self._read_raw("Operating_Mode") != 0:
            raise HapticHardwareError("selected motor must already be in position mode")
        self.check_health()
        self._original_torque_limit = self._read_raw("Torque_Limit")
        current = float(current_action[self.config.action_index])
        if not math.isfinite(current):
            raise HapticHardwareError("selected motor position is non-finite")
        try:
            self._write("Torque_Limit", 0, normalize=False)
            self._write("Goal_Position", current, normalize=True)
            self.bus.enable_torque([self.config.motor], num_retry=2)
            self._armed = True
        except Exception:
            self.emergency_release()
            raise

    def apply(
        self,
        command: HapticImpedanceCommand,
        current_action: Sequence[float],
    ) -> None:
        if not self._armed:
            raise HapticHardwareError("renderer is not armed")
        if len(current_action) != 12:
            raise HapticHardwareError("current action must contain 12 values")
        index = self.config.action_index
        torque_limit = int(command.torque_limit_raw[index]) if command.enabled else 0
        torque_limit = min(torque_limit, self.config.max_torque_limit_raw)
        if torque_limit <= 0:
            self._write("Torque_Limit", 0, normalize=False)
            self.output_commands += 1
            return
        target = float(current_action[index]) + float(command.position_offset_deg[index])
        if not math.isfinite(target):
            raise HapticHardwareError("computed goal position is non-finite")
        # Goal changes are already slew-limited by SafeHapticController.  Write
        # the small goal first, then expose it through the bounded torque limit.
        self._write("Goal_Position", target, normalize=True)
        self._write("Torque_Limit", torque_limit, normalize=False)
        self.output_commands += 1

    def emergency_release(self) -> None:
        """Best-effort fail-zero path that never hides the primary exception."""
        self.release_attempted = True
        try:
            self._write("Torque_Limit", 0, normalize=False)
        except Exception:
            pass
        try:
            self.bus.disable_torque([self.config.motor], num_retry=2)
        except Exception:
            pass
        self._armed = False

    def close(self) -> None:
        original = self._original_torque_limit
        if not self._armed and original is None:
            # Intervention waiting is strictly read-only. If arming never
            # started there is nothing to release or restore.
            return
        self.emergency_release()
        # A normal close is an evidence path, not merely a best effort: verify
        # that torque is disabled and its limit is zero before restoring SRAM.
        if self._read_raw("Torque_Enable") != 0:
            raise HapticHardwareError("torque remained enabled after release")
        if self._read_raw("Torque_Limit") != 0:
            raise HapticHardwareError("torque limit did not reach zero on release")
        self.release_verified = True
        if original is not None:
            # Restore the pre-run SRAM limit only after torque is disabled.
            self._write("Torque_Limit", original, normalize=False)
            if self._read_raw("Torque_Limit") != original:
                raise HapticHardwareError("prior torque limit was not restored")
            self.restore_verified = True
        self._original_torque_limit = None
