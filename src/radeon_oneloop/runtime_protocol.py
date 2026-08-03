"""CPU-edge message validation and fail-closed watchdog."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from .contracts import ActionLimits


class RuntimeState(str, Enum):
    DISARMED = "disarmed"
    ARMED = "armed"
    ESTOP = "estop"


@dataclass(frozen=True)
class ObservationEnvelope:
    sequence_id: int
    captured_monotonic_ns: int
    state: tuple[float, ...]


@dataclass(frozen=True)
class ActionEnvelope:
    sequence_id: int
    observation_sequence_id: int
    generated_monotonic_ns: int
    chunk: tuple[tuple[float, ...], ...]


class SafetyController:
    def __init__(
        self,
        limits: ActionLimits,
        *,
        observation_timeout_ms: float = 250.0,
        command_timeout_ms: float = 250.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.limits = limits
        self.observation_timeout_ns = int(observation_timeout_ms * 1_000_000)
        self.command_timeout_ns = int(command_timeout_ms * 1_000_000)
        self.clock_ns = clock_ns
        self.state = RuntimeState.DISARMED
        self._last_action: tuple[float, ...] | None = None
        self._last_sequence_id = -1

    def arm(self) -> None:
        if self.state is RuntimeState.ESTOP:
            raise RuntimeError("estop is latched; create a new controller after physical reset")
        self.state = RuntimeState.ARMED

    def estop(self) -> None:
        self.state = RuntimeState.ESTOP

    def validate(self, observation: ObservationEnvelope, action: ActionEnvelope) -> tuple[tuple[float, ...], ...]:
        if self.state is not RuntimeState.ARMED:
            raise RuntimeError(f"controller is not armed: {self.state.value}")
        now = self.clock_ns()
        if action.sequence_id <= self._last_sequence_id:
            self.estop()
            raise RuntimeError("non-monotonic action sequence")
        if action.observation_sequence_id != observation.sequence_id:
            self.estop()
            raise RuntimeError("action does not correspond to the current observation")
        if now - observation.captured_monotonic_ns > self.observation_timeout_ns:
            self.estop()
            raise TimeoutError("observation timeout")
        if now - action.generated_monotonic_ns > self.command_timeout_ns:
            self.estop()
            raise TimeoutError("command timeout")
        if not action.chunk:
            self.estop()
            raise RuntimeError("empty action chunk")

        checked = []
        previous: Sequence[float] | None = self._last_action
        try:
            for item in action.chunk:
                value = self.limits.validate(item, previous)
                checked.append(value)
                previous = value
        except Exception:
            self.estop()
            raise
        self._last_action = checked[-1]
        self._last_sequence_id = action.sequence_id
        return tuple(checked)

