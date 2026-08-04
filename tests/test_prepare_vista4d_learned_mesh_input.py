from __future__ import annotations

import numpy as np
import pytest

from gaussian.prepare_vista4d_learned_mesh_input import validate_orbit_contract


def _manifest() -> dict:
    return {
        "orbit": {
            "frames": 49,
            "image_size_wh": [672, 384],
            "endpoint_duplicate": False,
            "camera_schedule": "vista4d_unique_49_frame_level_orbit",
            "render_camera_model": "PINHOLE_OPENCV_fixed_intrinsic",
        }
    }


def _cameras() -> dict[str, np.ndarray]:
    return {
        "azimuth_deg": np.arange(49, dtype=np.float64) * 360.0 / 49.0,
        "cam_c2w": np.repeat(np.eye(4)[None], 49, axis=0),
        "intrinsics": np.ones((49, 4), dtype=np.float64),
    }


def test_accepts_exact_unique_camera_schedule() -> None:
    validate_orbit_contract(_manifest(), _cameras())


def test_rejects_duplicated_endpoint() -> None:
    cameras = _cameras()
    cameras["azimuth_deg"] = np.linspace(0.0, 360.0, 49, endpoint=True)
    with pytest.raises(ValueError, match="azimuths are not exact"):
        validate_orbit_contract(_manifest(), cameras)


def test_rejects_orthographic_autofit_source() -> None:
    manifest = _manifest()
    manifest["orbit"]["render_camera_model"] = "ORTHOGRAPHIC_per_view_autofit"
    with pytest.raises(ValueError, match="fixed pinhole"):
        validate_orbit_contract(manifest, _cameras())
