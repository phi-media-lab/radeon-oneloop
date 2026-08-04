import math
import unittest

import numpy as np

from gaussian.object_colmap_export import (
    ObjectColmapError,
    rotation_matrix_to_colmap_qvec,
    scale_intrinsic,
    validate_pose_audit,
)


def qvec_to_rotation(qvec):
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class ObjectColmapGeometryTests(unittest.TestCase):
    def test_rotation_round_trip_uses_colmap_hamilton_order(self):
        angle = math.radians(73)
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        )
        qvec = rotation_matrix_to_colmap_qvec(rotation)
        np.testing.assert_allclose(qvec_to_rotation(qvec), rotation, atol=1e-10)
        self.assertGreaterEqual(qvec[0], 0)

    def test_reflection_is_rejected(self):
        reflection = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ObjectColmapError, "proper"):
            rotation_matrix_to_colmap_qvec(reflection)

    def test_intrinsics_scale_from_vggt_512_to_m1_1024(self):
        intrinsic = np.asarray([[500.0, 0.0, 256.0], [0.0, 510.0, 256.0], [0.0, 0.0, 1.0]])
        scaled = scale_intrinsic(intrinsic, (512, 512), (1024, 1024))
        np.testing.assert_allclose(
            scaled,
            [[1000.0, 0.0, 512.0], [0.0, 1020.0, 512.0], [0.0, 0.0, 1.0]],
        )

    def test_pose_audit_must_accept_the_exact_run(self):
        value = {
            "schema_version": "radeon_oneloop.object_pose_visual_audit.v1",
            "formal": False,
            "source_run_manifest_sha256": "a" * 64,
            "review": {"status": "accepted_pose_and_coarse_geometry_initializer"},
        }
        validate_pose_audit(value, expected_pose_run_sha256="a" * 64)
        value["review"]["status"] = "rejected"
        with self.assertRaisesRegex(ObjectColmapError, "not passed"):
            validate_pose_audit(value, expected_pose_run_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
