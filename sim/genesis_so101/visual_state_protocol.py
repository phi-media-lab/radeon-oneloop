"""Versioned UDP snapshot contract from authoritative sim to demo renderer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


SCHEMA_VERSION = "radeon_oneloop.visual_state.v1"
MAX_PACKET_BYTES = 4096


class VisualStateProtocolError(ValueError):
    pass


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise VisualStateProtocolError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise VisualStateProtocolError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class VisualStatePacket:
    sequence_id: int
    captured_monotonic_ns: int
    captured_unix_ns: int
    joint_positions_rad: tuple[float, ...]
    object_position_m: tuple[float, ...]
    object_quaternion_wxyz: tuple[float, ...]

    def validate(self) -> None:
        if self.sequence_id < 0:
            raise VisualStateProtocolError("sequence_id must be non-negative")
        if self.captured_monotonic_ns <= 0 or self.captured_unix_ns <= 0:
            raise VisualStateProtocolError("timestamps must be positive")
        if len(self.joint_positions_rad) != 12 or not all(
            math.isfinite(value) and abs(value) <= 2.0 * math.pi
            for value in self.joint_positions_rad
        ):
            raise VisualStateProtocolError("joint positions are outside the snapshot envelope")
        if len(self.object_position_m) != 3 or not all(
            math.isfinite(value) and abs(value) <= 10.0
            for value in self.object_position_m
        ):
            raise VisualStateProtocolError("object position is outside the snapshot envelope")
        if len(self.object_quaternion_wxyz) != 4 or not all(
            math.isfinite(value) for value in self.object_quaternion_wxyz
        ):
            raise VisualStateProtocolError("object quaternion must have four finite values")
        norm = math.sqrt(sum(value * value for value in self.object_quaternion_wxyz))
        if not 0.99 <= norm <= 1.01:
            raise VisualStateProtocolError("object quaternion is not normalized")


def encode_visual_state(packet: VisualStatePacket) -> bytes:
    packet.validate()
    document = {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": packet.sequence_id,
        "captured_monotonic_ns": packet.captured_monotonic_ns,
        "captured_unix_ns": packet.captured_unix_ns,
        "joint_positions_rad": list(packet.joint_positions_rad),
        "object_position_m": list(packet.object_position_m),
        "object_quaternion_wxyz": list(packet.object_quaternion_wxyz),
    }
    payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(payload) > MAX_PACKET_BYTES:
        raise VisualStateProtocolError("encoded snapshot exceeds packet limit")
    return payload


def decode_visual_state(payload: bytes) -> VisualStatePacket:
    if not payload or len(payload) > MAX_PACKET_BYTES:
        raise VisualStateProtocolError("invalid snapshot packet size")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VisualStateProtocolError(f"invalid snapshot JSON: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise VisualStateProtocolError("unsupported snapshot schema")
    required = {
        "schema_version",
        "sequence_id",
        "captured_monotonic_ns",
        "captured_unix_ns",
        "joint_positions_rad",
        "object_position_m",
        "object_quaternion_wxyz",
    }
    if set(document) != required:
        raise VisualStateProtocolError("snapshot fields do not match the frozen schema")
    packet = VisualStatePacket(
        sequence_id=int(document["sequence_id"]),
        captured_monotonic_ns=int(document["captured_monotonic_ns"]),
        captured_unix_ns=int(document["captured_unix_ns"]),
        joint_positions_rad=_finite_vector(
            document["joint_positions_rad"], 12, "joint_positions_rad"
        ),
        object_position_m=_finite_vector(
            document["object_position_m"], 3, "object_position_m"
        ),
        object_quaternion_wxyz=_finite_vector(
            document["object_quaternion_wxyz"], 4, "object_quaternion_wxyz"
        ),
    )
    packet.validate()
    return packet
