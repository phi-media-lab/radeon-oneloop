from __future__ import annotations

import pytest

from gaussian.hil_object_coverage import (
    parse_episode_spec,
    phase_frame_indexes,
    successful_episode_indexes,
)


def test_phase_frame_indexes_cover_episode_interior_and_ends() -> None:
    indexes = phase_frame_indexes(1_418, 12)
    assert len(indexes) == 12
    assert indexes[0] == 0
    assert indexes[-1] == 1_417
    assert all(right > left for left, right in zip(indexes, indexes[1:]))


def test_phase_frame_indexes_reject_invalid_request() -> None:
    with pytest.raises(ValueError, match="positive"):
        phase_frame_indexes(0, 12)
    with pytest.raises(ValueError, match="at least two"):
        phase_frame_indexes(10, 1)


def test_parse_episode_spec_is_sorted_and_deduplicated() -> None:
    assert parse_episode_spec("5,0-2,2,4") == (0, 1, 2, 4, 5)


def test_successful_episode_indexes_filters_failures() -> None:
    records = {
        4: {"episode_success": "failure"},
        2: {"episode_success": "success"},
        1: {"episode_success": "success"},
    }
    assert successful_episode_indexes(records) == (1, 2)
