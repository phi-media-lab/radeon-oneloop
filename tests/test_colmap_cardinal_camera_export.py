import numpy as np

from gaussian.colmap_cardinal_camera_export import qvec_to_rotation
from gaussian.object_colmap_export import rotation_matrix_to_colmap_qvec


def test_colmap_quaternion_round_trip() -> None:
    angle = np.deg2rad(41.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        qvec_to_rotation(rotation_matrix_to_colmap_qvec(rotation)), rotation, atol=1.0e-12
    )
