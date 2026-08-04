from __future__ import annotations

import argparse

import pytest

from gaussian.record_learned_mesh_orbit_review import (
    ACCEPTED,
    REJECTED,
    REQUIRED_CHECKS,
    parse_check,
)


def test_parse_check_accepts_only_named_boolean_checks() -> None:
    assert parse_check(f"{REQUIRED_CHECKS[0]}=true") == (REQUIRED_CHECKS[0], True)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_check("unknown=true")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_check(f"{REQUIRED_CHECKS[0]}=yes")


def test_decision_constants_are_unambiguous() -> None:
    assert ACCEPTED != REJECTED
    assert len(REQUIRED_CHECKS) == len(set(REQUIRED_CHECKS))
