"""Wire contract for streaming two physical SO-101 leaders into Genesis.

The sender and receiver may run on different hosts, so sender monotonic time is
used only to validate ordering and rate-of-change. Receiver watchdogs must use
the local packet-arrival clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from radeon_oneloop.contracts import ACTION_NAMES, require_vector


SCHEMA_VERSION = "radeon_oneloop.leader_action.v1"
HAPTIC_SCHEMA_VERSION = "radeon_oneloop.haptic_feedback.v1"
MAX_PACKET_BYTES = 4096
SO101_MODEL_ACTION_MIN = (-110.0, -100.0, -100.0, -95.0, -160.0, 0.0) * 2
SO101_MODEL_ACTION_MAX = (110.0, 100.0, 90.0, 95.0, 160.0, 100.0) * 2


class LiveProtocolError(ValueError):
    """Raised when a live-leader packet violates the wire contract."""


def clamp_action_to_model(action: tuple[float, ...]) -> tuple[float, ...]:
    """Clamp a valid leader action to the pinned SO-101 MJCF joint ranges."""
    try:
        checked = require_vector(action, label="leader action")
    except ValueError as exc:
        raise LiveProtocolError(str(exc)) from exc
    return tuple(
        min(max(value, minimum), maximum)
        for value, minimum, maximum in zip(
            checked, SO101_MODEL_ACTION_MIN, SO101_MODEL_ACTION_MAX, strict=True
        )
    )


@dataclass(frozen=True)
class LeaderActionPacket:
    sequence_id: int
    captured_monotonic_ns: int
    captured_unix_ns: int
    action: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise LiveProtocolError("sequence_id must be non-negative")
        if self.captured_monotonic_ns <= 0:
            raise LiveProtocolError("captured_monotonic_ns must be positive")
        if self.captured_unix_ns <= 0:
            raise LiveProtocolError("captured_unix_ns must be positive")
        try:
            checked = require_vector(self.action, label="leader action")
        except ValueError as exc:
            raise LiveProtocolError(str(exc)) from exc
        object.__setattr__(self, "action", checked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence_id": self.sequence_id,
            "captured_monotonic_ns": self.captured_monotonic_ns,
            "captured_unix_ns": self.captured_unix_ns,
            "action_names": list(ACTION_NAMES),
            "action": list(self.action),
        }


@dataclass(frozen=True)
class HapticFeedbackPacket:
    sequence_id: int
    captured_monotonic_ns: int
    captured_unix_ns: int
    joint_reaction_effort: tuple[float, ...]
    contact_force_n: tuple[float, float]

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise LiveProtocolError("sequence_id must be non-negative")
        if self.captured_monotonic_ns <= 0 or self.captured_unix_ns <= 0:
            raise LiveProtocolError("haptic timestamps must be positive")
        try:
            efforts = require_vector(
                self.joint_reaction_effort, label="joint reaction effort"
            )
        except ValueError as exc:
            raise LiveProtocolError(str(exc)) from exc
        forces = tuple(float(value) for value in self.contact_force_n)
        if len(forces) != 2 or any(
            not math.isfinite(value) or value < 0.0 for value in forces
        ):
            raise LiveProtocolError(
                "contact_force_n must contain two finite non-negative values"
            )
        if any(abs(value) > 100.0 for value in efforts):
            raise LiveProtocolError("joint reaction effort exceeds 100-unit sanity bound")
        if any(value > 10_000.0 for value in forces):
            raise LiveProtocolError("contact force exceeds 10000 N sanity bound")
        object.__setattr__(self, "joint_reaction_effort", efforts)
        object.__setattr__(self, "contact_force_n", forces)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HAPTIC_SCHEMA_VERSION,
            "sequence_id": self.sequence_id,
            "captured_monotonic_ns": self.captured_monotonic_ns,
            "captured_unix_ns": self.captured_unix_ns,
            "effort_names": list(ACTION_NAMES),
            "joint_reaction_effort": list(self.joint_reaction_effort),
            "contact_force_n": list(self.contact_force_n),
        }


def encode_packet(packet: LeaderActionPacket) -> bytes:
    payload = json.dumps(
        packet.as_dict(), ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > MAX_PACKET_BYTES:
        raise LiveProtocolError(f"packet exceeds {MAX_PACKET_BYTES} bytes")
    return payload


def decode_packet(payload: bytes) -> LeaderActionPacket:
    if not payload:
        raise LiveProtocolError("empty packet")
    if len(payload) > MAX_PACKET_BYTES:
        raise LiveProtocolError(f"packet exceeds {MAX_PACKET_BYTES} bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveProtocolError(f"invalid JSON packet: {exc}") from exc
    if not isinstance(value, dict):
        raise LiveProtocolError("packet root must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise LiveProtocolError(f"unsupported schema_version: {value.get('schema_version')!r}")
    if tuple(value.get("action_names", ())) != ACTION_NAMES:
        raise LiveProtocolError("action_names do not match the frozen 12-DoF contract")
    try:
        return LeaderActionPacket(
            sequence_id=int(value["sequence_id"]),
            captured_monotonic_ns=int(value["captured_monotonic_ns"]),
            captured_unix_ns=int(value["captured_unix_ns"]),
            action=tuple(float(item) for item in value["action"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveProtocolError(f"invalid packet fields: {exc}") from exc


def encode_haptic_packet(packet: HapticFeedbackPacket) -> bytes:
    payload = json.dumps(
        packet.as_dict(), ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > MAX_PACKET_BYTES:
        raise LiveProtocolError(f"packet exceeds {MAX_PACKET_BYTES} bytes")
    return payload


def decode_haptic_packet(payload: bytes) -> HapticFeedbackPacket:
    if not payload:
        raise LiveProtocolError("empty packet")
    if len(payload) > MAX_PACKET_BYTES:
        raise LiveProtocolError(f"packet exceeds {MAX_PACKET_BYTES} bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveProtocolError(f"invalid JSON packet: {exc}") from exc
    if not isinstance(value, dict):
        raise LiveProtocolError("packet root must be an object")
    if value.get("schema_version") != HAPTIC_SCHEMA_VERSION:
        raise LiveProtocolError(
            f"unsupported haptic schema_version: {value.get('schema_version')!r}"
        )
    if tuple(value.get("effort_names", ())) != ACTION_NAMES:
        raise LiveProtocolError("effort_names do not match the frozen 12-DoF contract")
    try:
        return HapticFeedbackPacket(
            sequence_id=int(value["sequence_id"]),
            captured_monotonic_ns=int(value["captured_monotonic_ns"]),
            captured_unix_ns=int(value["captured_unix_ns"]),
            joint_reaction_effort=tuple(
                float(item) for item in value["joint_reaction_effort"]
            ),
            contact_force_n=tuple(float(item) for item in value["contact_force_n"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveProtocolError(f"invalid haptic packet fields: {exc}") from exc


class LeaderActionGate:
    """Fail-closed sequence, range, and sender-rate validation."""

    def __init__(
        self,
        *,
        body_limit_deg: float = 180.0,
        body_velocity_limit_deg_s: float = 900.0,
        gripper_velocity_limit_pct_s: float = 500.0,
        minimum_sender_dt_s: float = 1.0 / 240.0,
        maximum_sender_dt_s: float = 0.5,
    ) -> None:
        for name, value in (
            ("body_limit_deg", body_limit_deg),
            ("body_velocity_limit_deg_s", body_velocity_limit_deg_s),
            ("gripper_velocity_limit_pct_s", gripper_velocity_limit_pct_s),
            ("minimum_sender_dt_s", minimum_sender_dt_s),
            ("maximum_sender_dt_s", maximum_sender_dt_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if minimum_sender_dt_s > maximum_sender_dt_s:
            raise ValueError("minimum_sender_dt_s must not exceed maximum_sender_dt_s")
        self.body_limit_deg = float(body_limit_deg)
        self.body_velocity_limit_deg_s = float(body_velocity_limit_deg_s)
        self.gripper_velocity_limit_pct_s = float(gripper_velocity_limit_pct_s)
        self.minimum_sender_dt_s = float(minimum_sender_dt_s)
        self.maximum_sender_dt_s = float(maximum_sender_dt_s)
        self._previous: LeaderActionPacket | None = None

    @property
    def previous(self) -> LeaderActionPacket | None:
        return self._previous

    def sender_gap_s(self, packet: LeaderActionPacket) -> float | None:
        """Return the sender-clock gap without mutating gate state."""
        if self._previous is None:
            return None
        return (
            packet.captured_monotonic_ns - self._previous.captured_monotonic_ns
        ) / 1_000_000_000.0

    def rebase(self, packet: LeaderActionPacket) -> LeaderActionPacket:
        """Accept an absolute-range-checked packet as a new rate baseline.

        This is intentionally explicit: a receiver may call it after a proven
        stream hiatus so that a sender-time gap cannot permanently latch the
        velocity gate closed. Sequence ordering remains fail-closed.
        """
        previous = self._previous
        if previous is not None and packet.sequence_id <= previous.sequence_id:
            raise LiveProtocolError(
                f"non-monotonic sequence_id {packet.sequence_id} after "
                f"{previous.sequence_id}"
            )
        self._previous = None
        try:
            return self.accept(packet)
        except Exception:
            self._previous = previous
            raise

    def accept(self, packet: LeaderActionPacket) -> LeaderActionPacket:
        for arm_offset in (0, 6):
            for joint_offset in range(5):
                index = arm_offset + joint_offset
                value = packet.action[index]
                if abs(value) > self.body_limit_deg:
                    raise LiveProtocolError(
                        f"action[{index}]={value:.6f} exceeds +/-{self.body_limit_deg:.6f} degrees"
                    )
            gripper_index = arm_offset + 5
            gripper = packet.action[gripper_index]
            if not 0.0 <= gripper <= 100.0:
                raise LiveProtocolError(
                    f"action[{gripper_index}]={gripper:.6f} outside gripper range [0, 100]"
                )

        previous = self._previous
        if previous is not None:
            if packet.sequence_id <= previous.sequence_id:
                raise LiveProtocolError(
                    f"non-monotonic sequence_id {packet.sequence_id} after {previous.sequence_id}"
                )
            sender_dt_s = self.sender_gap_s(packet)
            assert sender_dt_s is not None
            if sender_dt_s <= 0:
                raise LiveProtocolError("sender monotonic timestamp did not increase")
            if sender_dt_s > self.maximum_sender_dt_s:
                raise LiveProtocolError(
                    f"sender gap {sender_dt_s:.6f}s exceeds {self.maximum_sender_dt_s:.6f}s"
                )
            rate_dt_s = max(sender_dt_s, self.minimum_sender_dt_s)
            for index, (current, old) in enumerate(zip(packet.action, previous.action, strict=True)):
                velocity_limit = (
                    self.gripper_velocity_limit_pct_s
                    if index in (5, 11)
                    else self.body_velocity_limit_deg_s
                )
                velocity = abs(current - old) / rate_dt_s
                if velocity > velocity_limit:
                    raise LiveProtocolError(
                        f"action[{index}] velocity {velocity:.3f} exceeds {velocity_limit:.3f}"
                    )

        self._previous = packet
        return packet
