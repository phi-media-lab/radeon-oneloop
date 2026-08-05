from __future__ import annotations

import numpy as np
import pytest

from gaussian.seva_full_geometry_colmap import (
    SevaFullGeometryError,
    required_support_count,
    support_visual_hull_surface,
)


def test_required_support_count_is_inclusive_and_conservative() -> None:
    assert required_support_count(49, 0.90) == 45
    assert required_support_count(8, 1.0) == 8
    with pytest.raises(SevaFullGeometryError):
        required_support_count(3, 0.90)
    with pytest.raises(SevaFullGeometryError):
        required_support_count(49, 0.50)


def test_support_hull_uses_votes_instead_of_one_fixed_proxy_shape() -> None:
    intrinsic = [[10.0, 0.0, 16.0], [0.0, 10.0, 16.0], [0.0, 0.0, 1.0]]
    w2c = np.eye(4)
    w2c[2, 3] = 1.0
    camera = {
        "intrinsic_3x3": intrinsic,
        "world_to_camera_opencv_4x4": w2c.tolist(),
    }
    cameras = [camera] * 8
    masks = [np.ones((32, 32), dtype=np.uint8) * 255 for _ in cameras]
    # One noisy view rejects everything.  A 7/8 threshold must retain the
    # consensus volume, whereas a strict 8/8 threshold must reject it.
    masks[-1] = np.zeros((32, 32), dtype=np.uint8)

    surface, audit = support_visual_hull_surface(
        cameras,
        masks,
        half_extents_m=(0.2, 0.2, 0.2),
        resolution=32,
        minimum_support_fraction=0.875,
    )

    assert len(surface) == 6 * 32 * 32 - 12 * 32 + 8
    assert audit["minimum_support_views"] == 7
    assert audit["surface_support_views_min"] == 7
    with pytest.raises(SevaFullGeometryError, match="empty support visual hull"):
        support_visual_hull_surface(
            cameras,
            masks,
            half_extents_m=(0.2, 0.2, 0.2),
            resolution=32,
            minimum_support_fraction=1.0,
        )
