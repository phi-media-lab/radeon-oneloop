"""Pure safety kernel for the proposed SO-101 leader haptic controller.

This module never opens a serial port. It converts monitored Genesis reaction
signals into bounded impedance targets that a separately reviewed hardware
adapter may consume after an explicit arm/estop handshake.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .live_protocol import HapticFeedbackPacket


@dataclass(frozen=True)
class HapticSafetyConfig:
    watchdog_ms: float = 100.0
    contact_deadband_n: float = 0.5
    simulated_effort_full_scale: float = 3.35
    max_normalized_effort: float = 0.20
    max_position_offset_deg: float = 3.0
    max_torque_limit_raw: int = 80
    max_normalized_slew_per_update: float = 0.025

    def __post_init__(self) -> None:
        for name, value in (
            ("watchdog_ms", self.watchdog_ms),
            ("contact_deadband_n", self.contact_deadband_n),
            ("simulated_effort_full_scale", self.simulated_effort_full_scale),
            ("max_normalized_effort", self.max_normalized_effort),
            ("max_position_offset_deg", self.max_position_offset_deg),
            ("max_normalized_slew_per_update", self.max_normalized_slew_per_update),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.max_torque_limit_raw <= 1000:
            raise ValueError("max_torque_limit_raw must be between 1 and 1000")


@dataclass(frozen=True)
class HapticImpedanceCommand:
    enabled: bool
    position_offset_deg: tuple[float, ...]
    torque_limit_raw: tuple[int, ...]
    reason: str


class SafeHapticController:
    """Fail-zero, slew-limited haptic command generator."""

    def __init__(self, config: HapticSafetyConfig = HapticSafetyConfig()) -> None:
        self.config = config
        self._armed = False
        self._estop_latched = False
        self._last_arrival_ns: int | None = None
        self._normalized = (0.0,) * 12

    def arm(self, *, physical_estop_confirmed: bool) -> None:
        if not physical_estop_confirmed:
            raise RuntimeError("physical emergency stop must be confirmed before arming")
        if self._estop_latched:
            raise RuntimeError("haptic estop is latched; create a new controller after reset")
        self._armed = True

    def latch_estop(self) -> None:
        self._estop_latched = True
        self._armed = False
        self._normalized = (0.0,) * 12

    def _disabled(self, reason: str) -> HapticImpedanceCommand:
        self._normalized = (0.0,) * 12
        return HapticImpedanceCommand(False, (0.0,) * 12, (0,) * 12, reason)

    def update(
        self,
        packet: HapticFeedbackPacket,
        *,
        arrival_monotonic_ns: int,
    ) -> HapticImpedanceCommand:
        if self._estop_latched:
            return self._disabled("estop_latched")
        if not self._armed:
            return self._disabled("not_armed")
        if arrival_monotonic_ns <= 0:
            raise ValueError("arrival_monotonic_ns must be positive")
        self._last_arrival_ns = arrival_monotonic_ns

        target = []
        for index, effort in enumerate(packet.joint_reaction_effort):
            arm_index = 0 if index < 6 else 1
            if packet.contact_force_n[arm_index] < self.config.contact_deadband_n:
                value = 0.0
            else:
                value = effort / self.config.simulated_effort_full_scale
                value = min(
                    max(value, -self.config.max_normalized_effort),
                    self.config.max_normalized_effort,
                )
            old = self._normalized[index]
            delta = min(
                max(
                    value - old,
                    -self.config.max_normalized_slew_per_update,
                ),
                self.config.max_normalized_slew_per_update,
            )
            target.append(old + delta)
        self._normalized = tuple(target)
        return HapticImpedanceCommand(
            enabled=True,
            position_offset_deg=tuple(
                value * self.config.max_position_offset_deg
                for value in self._normalized
            ),
            torque_limit_raw=tuple(
                self.config.max_torque_limit_raw if abs(value) > 0.0 else 0
                for value in self._normalized
            ),
            reason="contact_feedback",
        )

    def watchdog(self, *, now_monotonic_ns: int) -> HapticImpedanceCommand | None:
        if self._last_arrival_ns is None:
            return self._disabled("no_feedback") if self._armed else None
        age_ms = (now_monotonic_ns - self._last_arrival_ns) / 1_000_000.0
        if age_ms > self.config.watchdog_ms:
            self._armed = False
            return self._disabled("feedback_timeout")
        return None
