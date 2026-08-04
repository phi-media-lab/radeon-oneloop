"""Fail-safe five-joint renderer for the staged single-arm haptic gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .haptic_hardware import ARM_MOTORS, BENCH_MOTORS, HapticHardwareError
from .haptic_safety import HapticImpedanceCommand


@dataclass(frozen=True)
class HapticArmConfig:
    side: str
    max_torque_limit_raw: int = 20
    max_abs_current_raw: int = 150
    max_temperature_c: int = 45
    min_voltage_raw: int = 60
    max_voltage_raw: int = 84

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if not 1 <= self.max_torque_limit_raw <= 20:
            raise ValueError("single-arm torque limit must be in [1, 20]")
        if self.max_abs_current_raw <= 0 or self.max_temperature_c <= 0:
            raise ValueError("current and temperature bounds must be positive")
        if not 0 < self.min_voltage_raw < self.max_voltage_raw:
            raise ValueError("invalid voltage bounds")

    @property
    def action_indices(self) -> tuple[int, ...]:
        side_offset = 0 if self.side == "left" else len(ARM_MOTORS)
        return tuple(side_offset + index for index in range(len(BENCH_MOTORS)))


class FeetechHapticArmRenderer:
    """Render bounded impedance on five joints and verify fail-zero shutdown."""

    def __init__(self, bus: Any, config: HapticArmConfig) -> None:
        self.bus = bus
        self.config = config
        self._armed = False
        self._original_torque_limits: dict[str, int] = {}
        self.output_commands = 0
        self.peak_abs_current_raw = 0
        self.peak_temperature_c = 0
        self.release_attempted = False
        self.release_verified = False
        self.restore_verified = False

    @property
    def armed(self) -> bool:
        return self._armed

    def _read_raw(self, register: str, motor: str) -> int:
        return int(
            self.bus.read(register, motor, normalize=False, num_retry=2)
        )

    def _write_raw(self, register: str, motor: str, value: int) -> None:
        self.bus.write(
            register, motor, value, normalize=False, num_retry=2
        )

    def check_health(self) -> dict[str, dict[str, int]]:
        snapshots = {
            "present_current_raw": self.bus.sync_read(
                "Present_Current", list(BENCH_MOTORS), normalize=False, num_retry=2
            ),
            "present_temperature_c": self.bus.sync_read(
                "Present_Temperature", list(BENCH_MOTORS), normalize=False, num_retry=2
            ),
            "present_voltage_raw": self.bus.sync_read(
                "Present_Voltage", list(BENCH_MOTORS), normalize=False, num_retry=2
            ),
            "status": self.bus.sync_read(
                "Status", list(BENCH_MOTORS), normalize=False, num_retry=2
            ),
        }
        result: dict[str, dict[str, int]] = {}
        for motor in BENCH_MOTORS:
            current = int(snapshots["present_current_raw"][motor])
            temperature = int(snapshots["present_temperature_c"][motor])
            voltage = int(snapshots["present_voltage_raw"][motor])
            status = int(snapshots["status"][motor])
            self.peak_abs_current_raw = max(
                self.peak_abs_current_raw, abs(current)
            )
            self.peak_temperature_c = max(
                self.peak_temperature_c, temperature
            )
            if abs(current) > self.config.max_abs_current_raw:
                raise HapticHardwareError(
                    f"{motor} current limit exceeded: {current}"
                )
            if temperature > self.config.max_temperature_c:
                raise HapticHardwareError(
                    f"{motor} temperature limit exceeded: {temperature} C"
                )
            if not self.config.min_voltage_raw <= voltage <= self.config.max_voltage_raw:
                raise HapticHardwareError(
                    f"{motor} voltage outside bounds: {voltage}"
                )
            if status != 0:
                raise HapticHardwareError(f"{motor} status is nonzero: {status}")
            result[motor] = {
                "present_current_raw": current,
                "present_temperature_c": temperature,
                "present_voltage_raw": voltage,
                "status": status,
            }
        return result

    def arm(
        self, current_action: Sequence[float], *, physical_estop_confirmed: bool
    ) -> None:
        if self._armed:
            raise HapticHardwareError("renderer is already armed")
        if not physical_estop_confirmed:
            raise HapticHardwareError(
                "a reachable physical power cut/emergency stop must be confirmed"
            )
        if len(current_action) != 12:
            raise HapticHardwareError("current action must contain 12 values")
        for motor in BENCH_MOTORS:
            if self._read_raw("Torque_Enable", motor) != 0:
                raise HapticHardwareError(
                    f"{motor} torque must be disabled before arming"
                )
            if self._read_raw("Operating_Mode", motor) != 0:
                raise HapticHardwareError(
                    f"{motor} must already be in position mode"
                )
        self.check_health()
        self._original_torque_limits = {
            motor: self._read_raw("Torque_Limit", motor)
            for motor in BENCH_MOTORS
        }
        try:
            for motor, index in zip(
                BENCH_MOTORS, self.config.action_indices, strict=True
            ):
                current = float(current_action[index])
                if not math.isfinite(current):
                    raise HapticHardwareError(
                        f"{motor} current position is non-finite"
                    )
                self._write_raw("Torque_Limit", motor, 0)
                self.bus.write(
                    "Goal_Position",
                    motor,
                    current,
                    normalize=True,
                    num_retry=2,
                )
            self.bus.enable_torque(list(BENCH_MOTORS), num_retry=2)
            for motor in BENCH_MOTORS:
                if self._read_raw("Torque_Enable", motor) != 1:
                    raise HapticHardwareError(f"{motor} did not enable torque")
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
        goals: dict[str, float] = {}
        limits: dict[str, int] = {}
        for motor, index in zip(
            BENCH_MOTORS, self.config.action_indices, strict=True
        ):
            torque_limit = (
                int(command.torque_limit_raw[index]) if command.enabled else 0
            )
            torque_limit = min(
                max(torque_limit, 0), self.config.max_torque_limit_raw
            )
            target = float(current_action[index])
            if torque_limit > 0:
                target += float(command.position_offset_deg[index])
            if not math.isfinite(target):
                raise HapticHardwareError(
                    f"computed {motor} goal position is non-finite"
                )
            goals[motor] = target
            limits[motor] = torque_limit
        # Goals precede exposure through the torque limit. Both transactions
        # are bounded by the prior frame's <=20/1000 limit if a packet is lost.
        self.bus.sync_write(
            "Goal_Position", goals, normalize=True, num_retry=2
        )
        self.bus.sync_write(
            "Torque_Limit", limits, normalize=False, num_retry=2
        )
        self.output_commands += 1

    def emergency_release(self) -> None:
        self.release_attempted = True
        try:
            self.bus.sync_write(
                "Torque_Limit",
                {motor: 0 for motor in BENCH_MOTORS},
                normalize=False,
                num_retry=2,
            )
        except Exception:
            pass
        try:
            self.bus.disable_torque(list(BENCH_MOTORS), num_retry=2)
        except Exception:
            pass
        self._armed = False

    def close(self) -> None:
        original = dict(self._original_torque_limits)
        if not self._armed and not original:
            # Intervention waiting is strictly read-only. If arming never
            # started there is nothing to release or restore.
            return
        self.emergency_release()
        for motor in BENCH_MOTORS:
            if self._read_raw("Torque_Enable", motor) != 0:
                raise HapticHardwareError(
                    f"{motor} torque remained enabled after release"
                )
            if self._read_raw("Torque_Limit", motor) != 0:
                raise HapticHardwareError(
                    f"{motor} torque limit did not reach zero on release"
                )
        self.release_verified = True
        for motor, torque_limit in original.items():
            self._write_raw("Torque_Limit", motor, torque_limit)
            if self._read_raw("Torque_Limit", motor) != torque_limit:
                raise HapticHardwareError(
                    f"{motor} prior torque limit was not restored"
                )
        self.restore_verified = len(original) == len(BENCH_MOTORS)
        self._original_torque_limits = {}
