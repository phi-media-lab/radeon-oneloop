"""Online task-evidence gate for the read-only dual-leader handover demo."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence


CONTACT_NONE = "none"
CONTACT_LEFT = "left_only"
CONTACT_DUAL = "dual"
CONTACT_RIGHT = "right_only"
CONTACT_LABELS = (CONTACT_NONE, CONTACT_LEFT, CONTACT_DUAL, CONTACT_RIGHT)


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))


@dataclass
class HandoverTaskTracker:
    start_position_m: tuple[float, float, float]
    contact_deadband_n: float = 0.5
    persistence_samples: int = 3
    minimum_samples: int = 30
    minimum_transfer_displacement_m: float = 0.12
    minimum_object_center_z_m: float = 0.43
    _last_position: tuple[float, float, float] = field(init=False)
    _sample_count: int = field(default=0, init=False)
    _path_length_m: float = field(default=0.0, init=False)
    _maximum_displacement_m: float = field(default=0.0, init=False)
    _minimum_position_m: list[float] = field(init=False)
    _maximum_position_m: list[float] = field(init=False)
    _contact_counts: dict[str, int] = field(init=False)
    _run_label: str | None = field(default=None, init=False)
    _run_samples: int = field(default=0, init=False)
    _phase: str = field(default="waiting_left_grasp", init=False)
    _events: list[dict[str, object]] = field(default_factory=list, init=False)
    _initial_no_contact_observed: bool = field(default=False, init=False)
    _sequence_violations: list[dict[str, object]] = field(default_factory=list, init=False)
    _finite: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if len(self.start_position_m) != 3 or not all(
            math.isfinite(value) for value in self.start_position_m
        ):
            raise ValueError("start position must contain three finite values")
        if not math.isfinite(self.contact_deadband_n) or self.contact_deadband_n <= 0:
            raise ValueError("contact deadband must be positive")
        if self.persistence_samples < 1 or self.minimum_samples < self.persistence_samples:
            raise ValueError("invalid task persistence/sample contract")
        self._last_position = tuple(float(value) for value in self.start_position_m)
        self._minimum_position_m = list(self._last_position)
        self._maximum_position_m = list(self._last_position)
        self._contact_counts = {label: 0 for label in CONTACT_LABELS}

    def _contact_label(self, force_n: Sequence[float]) -> str:
        if len(force_n) != 2 or not all(math.isfinite(float(value)) for value in force_n):
            self._finite = False
            return CONTACT_NONE
        left = float(force_n[0]) >= self.contact_deadband_n
        right = float(force_n[1]) >= self.contact_deadband_n
        if left and right:
            return CONTACT_DUAL
        if left:
            return CONTACT_LEFT
        if right:
            return CONTACT_RIGHT
        return CONTACT_NONE

    def update(
        self,
        *,
        sample_index: int,
        object_position_m: Sequence[float],
        contact_force_n: Sequence[float],
    ) -> None:
        if sample_index < 0:
            raise ValueError("sample index must be nonnegative")
        if len(object_position_m) != 3 or not all(
            math.isfinite(float(value)) for value in object_position_m
        ):
            self._finite = False
            return
        position = tuple(float(value) for value in object_position_m)
        self._sample_count += 1
        self._path_length_m += _distance(self._last_position, position)
        self._maximum_displacement_m = max(
            self._maximum_displacement_m,
            _distance(self.start_position_m, position),
        )
        for axis, value in enumerate(position):
            self._minimum_position_m[axis] = min(self._minimum_position_m[axis], value)
            self._maximum_position_m[axis] = max(self._maximum_position_m[axis], value)
        self._last_position = position

        label = self._contact_label(contact_force_n)
        self._contact_counts[label] += 1
        if label == CONTACT_NONE and not self._events:
            self._initial_no_contact_observed = True
        if label == self._run_label:
            self._run_samples += 1
        else:
            self._run_label = label
            self._run_samples = 1
        if self._run_samples != self.persistence_samples:
            return

        event = None
        if self._phase == "waiting_left_grasp":
            if label == CONTACT_LEFT:
                event = "left_grasp"
                self._phase = "waiting_dual_handover"
            elif label in {CONTACT_DUAL, CONTACT_RIGHT}:
                self._sequence_violations.append(
                    {"phase": self._phase, "label": label, "sample_index": sample_index}
                )
        elif self._phase == "waiting_dual_handover":
            if label == CONTACT_DUAL:
                event = "dual_handover"
                self._phase = "waiting_left_release"
            elif label in {CONTACT_NONE, CONTACT_RIGHT}:
                self._sequence_violations.append(
                    {"phase": self._phase, "label": label, "sample_index": sample_index}
                )
        elif self._phase == "waiting_left_release":
            if label == CONTACT_RIGHT:
                event = "left_release_right_holds"
                self._phase = "complete"
            elif label in {CONTACT_NONE, CONTACT_LEFT}:
                self._sequence_violations.append(
                    {"phase": self._phase, "label": label, "sample_index": sample_index}
                )
        if event is not None:
            self._events.append(
                {
                    "event": event,
                    "sample_index": sample_index,
                    "object_position_m": list(position),
                }
            )

    def summary(
        self,
        *,
        target_position_m: Sequence[float],
        target_tolerance_m: float,
    ) -> dict[str, object]:
        if len(target_position_m) != 3 or not all(
            math.isfinite(float(value)) for value in target_position_m
        ):
            raise ValueError("target position must contain three finite values")
        if not math.isfinite(target_tolerance_m) or target_tolerance_m <= 0:
            raise ValueError("target tolerance must be positive")
        target = tuple(float(value) for value in target_position_m)
        target_distance = _distance(self._last_position, target)
        event_names = [str(item["event"]) for item in self._events]
        checks = {
            "finite_samples": self._finite,
            "minimum_sample_count": self._sample_count >= self.minimum_samples,
            "initial_no_contact_observed": self._initial_no_contact_observed,
            "ordered_left_grasp_dual_handover_left_release": event_names
            == ["left_grasp", "dual_handover", "left_release_right_holds"]
            and not self._sequence_violations,
            "minimum_transfer_displacement": self._maximum_displacement_m
            >= self.minimum_transfer_displacement_m,
            "target_reached": target_distance <= target_tolerance_m,
            "object_not_dropped_below_table_envelope": self._minimum_position_m[2]
            >= self.minimum_object_center_z_m,
        }
        return {
            "schema_version": "radeon_oneloop.handover_task_gate.v1",
            "accepted": all(checks.values()),
            "checks": checks,
            "required_sequence": [
                "approach_without_contact",
                "left_grasp",
                "dual_handover",
                "left_release_right_holds",
            ],
            "events": self._events,
            "sequence_violations": self._sequence_violations,
            "contact_deadband_n": self.contact_deadband_n,
            "persistence_samples": self.persistence_samples,
            "sample_count": self._sample_count,
            "contact_sample_counts": self._contact_counts,
            "start_position_m": list(self.start_position_m),
            "target_position_m": list(target),
            "final_position_m": list(self._last_position),
            "target_tolerance_m": target_tolerance_m,
            "final_target_distance_m": target_distance,
            "maximum_displacement_m": self._maximum_displacement_m,
            "path_length_m": self._path_length_m,
            "position_bounds_m": {
                "minimum": self._minimum_position_m,
                "maximum": self._maximum_position_m,
            },
            "physical_output_commands": False,
        }
