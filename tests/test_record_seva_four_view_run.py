import unittest

import numpy as np

from gaussian.record_seva_four_view_run import camera_error


def transforms(count: int = 53):
    return {
        "frames": [
            {"transform_matrix": (np.eye(4) + index * 1e-4).tolist()}
            for index in range(count)
        ]
    }


class SevaFourViewAuditTests(unittest.TestCase):
    def test_exact_cameras_have_zero_error(self):
        value = transforms()
        self.assertEqual(camera_error(value, value), 0.0)

    def test_detects_target_camera_drift(self):
        source = transforms()
        output = transforms()
        output["frames"][10]["transform_matrix"][0][3] += 0.01
        self.assertAlmostEqual(camera_error(source, output), 0.01)

    def test_accepts_seva_three_by_four_output_matrices(self):
        source = transforms()
        for frame in source["frames"]:
            frame["transform_matrix"][3] = [0.0, 0.0, 0.0, 1.0]
        output = transforms()
        for source_frame, output_frame in zip(source["frames"], output["frames"]):
            output_frame["transform_matrix"] = source_frame["transform_matrix"][:3]
        self.assertEqual(camera_error(source, output), 0.0)

    def test_rejects_wrong_frame_count(self):
        with self.assertRaisesRegex(ValueError, "4 inputs and 49 targets"):
            camera_error(transforms(52), transforms())


if __name__ == "__main__":
    unittest.main()
