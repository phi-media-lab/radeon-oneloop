import importlib.util

import numpy as np
import pytest

from gaussian.hybrid_pseudoview_colmap import (
    recover_vista4d_w2c,
    resize_and_composite_anchor,
    weak_perspective_equivalent_camera,
)
from gaussian.texture_learned_mesh_four_views import canonical_orbit_extrinsic


def test_vista4d_storage_round_trip_recovers_opencv_extrinsic() -> None:
    expected = canonical_orbit_extrinsic(37.0, distance_m=0.24)
    conversion = np.diag([-1.0, -1.0, 1.0, 1.0])
    stored = conversion @ np.linalg.inv(expected)
    recovered = recover_vista4d_w2c(stored[None])
    np.testing.assert_allclose(recovered[0], expected, atol=1.0e-12)


def test_anchor_resize_preserves_aspect_and_alpha() -> None:
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("OpenCV is required")
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[25:75, 25:75] = 255
    image, mask, report = resize_and_composite_anchor(rgb, alpha, (200, 100))
    assert image.shape == (100, 200, 3)
    assert mask.shape == (100, 200)
    assert report["uniform_scale"] == 1.0
    assert report["offset_xy"] == [50, 0]
    assert np.all(image[:, :50] == 255)
    assert mask[50, 100] == 255


def test_weak_perspective_camera_is_metric_and_declared_uncalibrated() -> None:
    vertices = np.asarray(
        [
            [-0.04, -0.02, -0.05],
            [0.04, -0.02, -0.05],
            [-0.04, 0.02, 0.05],
            [0.04, 0.02, 0.05],
        ]
    )
    mask = np.zeros((384, 672), dtype=np.uint8)
    mask[80:304, 240:432] = 255
    intrinsic, w2c, report = weak_perspective_equivalent_camera(
        vertices, mask, "front", distance_m=1.0
    )
    assert intrinsic.shape == (3, 3)
    assert w2c.shape == (4, 4)
    assert intrinsic[0, 0] > 0
    assert report["camera_bound_observation"] is False
    assert report["model"].startswith("PINHOLE_weak_perspective")
