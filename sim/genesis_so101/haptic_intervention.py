"""Pure stable-safe-pose gate for a same-process HIL transition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .haptic_hardware import ARM_MOTORS, BENCH_MOTORS
from .live_protocol import SO101_MODEL_ACTION_MAX, SO101_MODEL_ACTION_MIN


@dataclass(frozen=True)
class StableSafePoseConfig:
    side: str
    motors: tuple[str, ...] = BENCH_MOTORS
    hold_s: float = 0.4
    max_span_deg: float = 2.0
    model_limit_margin_deg: float = 5.0
    max_position_offset_deg: float = 0.5

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if (
            not self.motors
            or len(set(self.motors)) != len(self.motors)
            or any(motor not in BENCH_MOTORS for motor in self.motors)
        ):
            raise ValueError("motors must be a unique non-empty non-gripper subset")
        if not 0.1 <= self.hold_s <= 2.0:
            raise ValueError("stable pose hold must be in [0.1, 2] seconds")
        if not 0.1 <= self.max_span_deg <= 5.0:
            raise ValueError("stable pose maximum span must be in [0.1, 5] degrees")
        if not 0.0 < self.model_limit_margin_deg <= 20.0:
            raise ValueError("model limit margin must be in (0, 20] degrees")
        if not 0.0 < self.max_position_offset_deg <= 1.0:
            raise ValueError("position offset must be in (0, 1] degrees")

    @property
    def action_indices(self) -> tuple[int, ...]:
        side_offset = 0 if self.side == "left" else len(ARM_MOTORS)
        return tuple(
            side_offset + ARM_MOTORS.index(motor) for motor in self.motors
        )

    @property
    def accepted_ranges_deg(self) -> dict[str, tuple[float, float]]:
        return {
            motor: (
                SO101_MODEL_ACTION_MIN[index]
                + self.model_limit_margin_deg
                + self.max_position_offset_deg,
                SO101_MODEL_ACTION_MAX[index]
                - self.model_limit_margin_deg
                - self.max_position_offset_deg,
            )
            for motor, index in zip(self.motors, self.action_indices, strict=True)
        }


class StableSafePoseGate:
    """Require a continuous, bounded-span safe pose before motor arming."""

    def __init__(self, config: StableSafePoseConfig) -> None:
        self.config = config
        self.samples_seen = 0
        self.safe_samples = 0
        self.reset_count = 0
        self._safe_since_ns: int | None = None
        self._minimum: list[float] | None = None
        self._maximum: list[float] | None = None
        self._latest: list[float] | None = None
        self._latest_position_checks: dict[str, bool] = {}
        self._last_reset_reason: str | None = None

    def _reset(self, reason: str) -> None:
        if self._safe_since_ns is not None:
            self.reset_count += 1
        self._safe_since_ns = None
        self._minimum = None
        self._maximum = None
        self._last_reset_reason = reason

    def update(self, action: Sequence[float], *, now_ns: int) -> bool:
        if now_ns <= 0:
            raise ValueError("now_ns must be positive")
        self.samples_seen += 1
        if len(action) != 12 or any(not math.isfinite(float(v)) for v in action):
            self._latest = None
            self._latest_position_checks = {}
            self._reset("invalid_action")
            return False

        selected = [float(action[index]) for index in self.config.action_indices]
        ranges = self.config.accepted_ranges_deg
        self._latest = selected
        self._latest_position_checks = {
            motor: lower <= value <= upper
            for motor, value, (lower, upper) in zip(
                self.config.motors, selected, ranges.values(), strict=True
            )
        }
        if not all(self._latest_position_checks.values()):
            self._reset("outside_bidirectional_margin")
            return False

        self.safe_samples += 1
        if self._safe_since_ns is None:
            self._safe_since_ns = now_ns
            self._minimum = selected.copy()
            self._maximum = selected.copy()
            self._last_reset_reason = None
        else:
            assert self._minimum is not None and self._maximum is not None
            self._minimum = [
                min(old, value)
                for old, value in zip(self._minimum, selected, strict=True)
            ]
            self._maximum = [
                max(old, value)
                for old, value in zip(self._maximum, selected, strict=True)
            ]
            if any(
                maximum - minimum > self.config.max_span_deg
                for minimum, maximum in zip(
                    self._minimum, self._maximum, strict=True
                )
            ):
                self.reset_count += 1
                self._safe_since_ns = now_ns
                self._minimum = selected.copy()
                self._maximum = selected.copy()
                self._last_reset_reason = "motion_span_exceeded"

        return self.stable_duration_s(now_ns) >= self.config.hold_s

    def stable_duration_s(self, now_ns: int) -> float:
        if self._safe_since_ns is None:
            return 0.0
        return max(0.0, (now_ns - self._safe_since_ns) / 1_000_000_000.0)

    def as_dict(self, *, now_ns: int) -> dict[str, object]:
        spans = None
        if self._minimum is not None and self._maximum is not None:
            spans = {
                motor: maximum - minimum
                for motor, minimum, maximum in zip(
                    self.config.motors,
                    self._minimum,
                    self._maximum,
                    strict=True,
                )
            }
        return {
            "schema_version": "radeon_oneloop.haptic_intervention_gate.v1",
            "mode": "same_process_stable_safe_pose",
            "side": self.config.side,
            "motors": list(self.config.motors),
            "trigger": "operator_attestation_plus_stable_safe_pose",
            "candidate_ready": (
                self.stable_duration_s(now_ns) >= self.config.hold_s
                and bool(self._latest_position_checks)
                and all(self._latest_position_checks.values())
            ),
            "hold_required_s": self.config.hold_s,
            "stable_duration_s": self.stable_duration_s(now_ns),
            "max_span_limit_deg": self.config.max_span_deg,
            "observed_span_deg_by_motor": spans,
            "accepted_ranges_deg_by_motor": {
                motor: list(bounds)
                for motor, bounds in self.config.accepted_ranges_deg.items()
            },
            "latest_position_deg_by_motor": (
                dict(zip(self.config.motors, self._latest, strict=True))
                if self._latest is not None
                else None
            ),
            "latest_position_checks": self._latest_position_checks,
            "samples_seen": self.samples_seen,
            "safe_samples": self.safe_samples,
            "reset_count": self.reset_count,
            "last_reset_reason": self._last_reset_reason,
            "serial_connection_preserved_for_arm": True,
        }
