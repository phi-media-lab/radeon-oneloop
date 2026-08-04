"""Shared read-only access helpers for calibrated SO-101 leaders."""

from __future__ import annotations

from typing import Any

from radeon_oneloop.contracts import ARM_JOINTS


def make_leader(port: str, arm_id: str) -> Any:
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    return SO101Leader(
        SO101LeaderConfig(port=port, id=arm_id, use_degrees=True)
    )


def connect_read_only(leader: Any) -> None:
    if not leader.calibration:
        raise RuntimeError(
            f"missing calibration for {leader.id}: {leader.calibration_fpath}"
        )
    # Deliberately bypass SO101Leader.connect(): that method configures motor
    # registers. A leader bridge only needs to open the bus and read positions.
    leader.bus.connect()
    if not leader.bus.is_calibrated:
        raise RuntimeError(
            f"motor calibration does not match {leader.calibration_fpath}; "
            "refusing to write calibration from the live reader"
        )


def read_arm(leader: Any) -> tuple[float, ...]:
    values = leader.get_action()
    return tuple(float(values[name]) for name in ARM_JOINTS)
