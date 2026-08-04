from pathlib import Path

import numpy as np
import pytest

from gaussian.export_observed_initialization import load_colmap_points


def test_load_colmap_points_reads_metric_xyz_and_rgb(tmp_path: Path) -> None:
    path = tmp_path / "points3D.txt"
    rows = ["# points"]
    for index in range(1, 1001):
        rows.append(f"{index} {index * 0.001} 0.0 -0.01 10 20 30 0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    xyz, rgb = load_colmap_points(path)
    assert xyz.shape == (1000, 3)
    assert rgb.shape == (1000, 3)
    np.testing.assert_array_equal(rgb[0], [10, 20, 30])


def test_load_colmap_points_rejects_too_few_points(tmp_path: Path) -> None:
    path = tmp_path / "points3D.txt"
    path.write_text("1 0 0 0 1 2 3 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="too few"):
        load_colmap_points(path)
