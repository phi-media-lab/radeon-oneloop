from __future__ import annotations

import numpy as np
import pytest

from gaussian.prune_generated_fill_visibility import VisibilityPruneError, visibility_count


def test_visibility_count_keeps_only_front_surface_at_same_pixel() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.002],
            [0.0, 0.0, 1.02],
            [4.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera = {
        "view_id": "front",
        "world_to_camera_opencv_4x4": np.eye(4).tolist(),
        "intrinsic_3x3": [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]],
    }
    masks = np.full((1, 5, 5), 255, dtype=np.uint8)
    counts, reports = visibility_count(
        positions, [camera], masks, surface_tolerance_m=0.004
    )
    np.testing.assert_array_equal(counts, [1, 1, 0, 0])
    assert reports[0]["front_visible_centers"] == 2


def test_visibility_count_respects_real_mask() -> None:
    positions = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    camera = {
        "world_to_camera_opencv_4x4": np.eye(4).tolist(),
        "intrinsic_3x3": np.eye(3).tolist(),
    }
    counts, _ = visibility_count(
        positions,
        [camera],
        np.zeros((1, 2, 2), dtype=np.uint8),
        surface_tolerance_m=0.0,
    )
    np.testing.assert_array_equal(counts, [0])


def test_visibility_count_validates_tolerance() -> None:
    with pytest.raises(VisibilityPruneError, match="tolerance"):
        visibility_count(
            np.zeros((1, 3)),
            [],
            np.zeros((0, 1, 1), dtype=np.uint8),
            surface_tolerance_m=-1.0,
        )
