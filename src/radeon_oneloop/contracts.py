"""Frozen observation, action, phase, and gripper contracts.

This module deliberately has no robotics or ML dependency so every machine can
validate the most safety-critical assumptions before a dataset or checkpoint is
opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


CAMERA_KEYS = (
    "observation.images.front_cam",
    "observation.images.hand_cam",
)

ARM_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

ACTION_NAMES = tuple(f"left_{name}" for name in ARM_JOINTS) + tuple(
    f"right_{name}" for name in ARM_JOINTS
)

PHASES = (
    "approach_and_grasp",
    "lift_and_align",
    "handover",
    "receive_and_transfer",
    "place_and_release",
)

STATE_KEY = "observation.state"
ACTION_KEY = "action"
FPS = 30
IMAGE_SHAPE_HWC = (480, 640, 3)


class ContractError(ValueError):
    """Raised when a dataset, observation, or action violates the frozen contract."""


def require_action_names(names: Sequence[str]) -> None:
    if tuple(names) != ACTION_NAMES:
        raise ContractError(f"action ordering mismatch: expected {ACTION_NAMES!r}, got {tuple(names)!r}")


def require_vector(values: Sequence[float], *, label: str = ACTION_KEY) -> tuple[float, ...]:
    if len(values) != len(ACTION_NAMES):
        raise ContractError(f"{label} must have {len(ACTION_NAMES)} values, got {len(values)}")
    out = tuple(float(v) for v in values)
    if any(v != v or v in (float("inf"), float("-inf")) for v in out):
        raise ContractError(f"{label} contains a non-finite value")
    return out


def gripper_percent_to_joint(value: float, *, joint_min: float, joint_max: float) -> float:
    """Map the real controller's closed=0/open=100 convention to a joint range."""
    value = float(value)
    if not 0.0 <= value <= 100.0:
        raise ContractError(f"gripper percent outside [0, 100]: {value}")
    if joint_max <= joint_min:
        raise ContractError("joint_max must be greater than joint_min")
    return joint_min + (value / 100.0) * (joint_max - joint_min)


def gripper_joint_to_percent(
    value: float,
    *,
    joint_min: float,
    joint_max: float,
    tolerance: float = 0.0,
) -> float:
    if joint_max <= joint_min:
        raise ContractError("joint_max must be greater than joint_min")
    value = float(value)
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ContractError("tolerance must be finite and non-negative")
    if not joint_min - tolerance <= value <= joint_max + tolerance:
        raise ContractError(f"gripper joint outside [{joint_min}, {joint_max}]: {value}")
    value = min(max(value, joint_min), joint_max)
    return 100.0 * (value - joint_min) / (joint_max - joint_min)


SO101_GRIPPER_MIN_RAD = math.radians(-10.0)
SO101_GRIPPER_MAX_RAD = math.radians(100.0)


def lerobot_arm_to_genesis(values: Sequence[float]) -> tuple[float, ...]:
    """Convert one LeRobot SO-101 arm (degrees + 0..100 gripper) to MJCF radians."""
    if len(values) != 6:
        raise ContractError(f"one SO-101 arm must have 6 values, got {len(values)}")
    joints = tuple(math.radians(float(value)) for value in values[:5])
    gripper = gripper_percent_to_joint(
        float(values[5]), joint_min=SO101_GRIPPER_MIN_RAD, joint_max=SO101_GRIPPER_MAX_RAD
    )
    return joints + (gripper,)


def genesis_arm_to_lerobot(
    values: Sequence[float],
    *,
    gripper_tolerance_rad: float = math.radians(0.5),
) -> tuple[float, ...]:
    """Convert one Genesis SO-101 arm back to LeRobot's physical-unit convention."""
    if len(values) != 6:
        raise ContractError(f"one SO-101 arm must have 6 values, got {len(values)}")
    joints = tuple(math.degrees(float(value)) for value in values[:5])
    gripper = gripper_joint_to_percent(
        float(values[5]),
        joint_min=SO101_GRIPPER_MIN_RAD,
        joint_max=SO101_GRIPPER_MAX_RAD,
        # The rigid solver may settle a position-controlled gripper a tiny
        # amount beyond its configured limit. Accept only this numerical
        # boundary layer, then clamp it back to the frozen 0..100 contract.
        tolerance=gripper_tolerance_rad,
    )
    return joints + (gripper,)


@dataclass(frozen=True)
class ActionLimits:
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    max_delta: tuple[float, ...]

    def __post_init__(self) -> None:
        for label, values in (
            ("minimum", self.minimum),
            ("maximum", self.maximum),
            ("max_delta", self.max_delta),
        ):
            require_vector(values, label=label)
        if any(lo >= hi for lo, hi in zip(self.minimum, self.maximum, strict=True)):
            raise ContractError("every action minimum must be lower than its maximum")
        if any(delta <= 0 for delta in self.max_delta):
            raise ContractError("every max_delta must be positive")

    def validate(self, action: Sequence[float], previous: Sequence[float] | None = None) -> tuple[float, ...]:
        values = require_vector(action)
        for index, (value, lo, hi) in enumerate(zip(values, self.minimum, self.maximum, strict=True)):
            if not lo <= value <= hi:
                raise ContractError(f"action[{index}]={value} outside [{lo}, {hi}]")
        if previous is not None:
            prior = require_vector(previous, label="previous action")
            for index, (value, old, delta) in enumerate(zip(values, prior, self.max_delta, strict=True)):
                if abs(value - old) > delta:
                    raise ContractError(
                        f"action[{index}] delta {abs(value - old):.6f} exceeds {delta:.6f}"
                    )
        return values


def assert_unique(values: Iterable[str], *, label: str) -> None:
    values = tuple(values)
    if len(values) != len(set(values)):
        raise ContractError(f"{label} contains duplicates")
