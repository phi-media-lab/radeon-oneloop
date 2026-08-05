from __future__ import annotations

import numpy as np
import pytest

from gaussian.seva_pseudoview_colmap import (
    ANCHOR_INDEX_BY_LABEL,
    SevaDatasetError,
    opengl_c2w_to_metric_opencv_w2c,
    select_generated_indices,
)


def test_generated_selection_is_even_unique_and_excludes_anchor_targets() -> None:
    selected = select_generated_indices(24)
    assert len(selected) == 24
    assert len(set(selected)) == 24
    assert not set(selected) & set(ANCHOR_INDEX_BY_LABEL.values())
    assert min(selected) >= 0 and max(selected) < 49
    with pytest.raises(SevaDatasetError):
        select_generated_indices(46)


def test_opengl_camera_conversion_resolves_metric_radius() -> None:
    c2w_gl = np.eye(4)
    c2w_gl[:3, 3] = [0.0, 2.0, 0.0]
    w2c_cv = opengl_c2w_to_metric_opencv_w2c(c2w_gl, 0.2)
    c2w_cv = np.linalg.inv(w2c_cv)
    assert np.linalg.norm(c2w_cv[:3, 3]) == pytest.approx(0.2)
    np.testing.assert_allclose(c2w_cv[:3, 0], c2w_gl[:3, 0])
    np.testing.assert_allclose(c2w_cv[:3, 1], -c2w_gl[:3, 1])
    np.testing.assert_allclose(c2w_cv[:3, 2], -c2w_gl[:3, 2])


def test_opengl_camera_conversion_accepts_native_seva_three_by_four() -> None:
    c2w_gl = np.eye(4)
    c2w_gl[:3, 3] = [0.0, 2.0, 0.0]
    expected = opengl_c2w_to_metric_opencv_w2c(c2w_gl, 0.2)
    actual = opengl_c2w_to_metric_opencv_w2c(c2w_gl[:3], 0.2)
    np.testing.assert_allclose(actual, expected)


def test_camera_conversion_rejects_zero_radius() -> None:
    with pytest.raises(SevaDatasetError, match="zero radius"):
        opengl_c2w_to_metric_opencv_w2c(np.eye(4), 0.2)
