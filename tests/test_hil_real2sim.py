import tempfile
import unittest
from pathlib import Path

from gaussian.colmap_workspace import parse_text_model

try:
    import numpy as np
except ImportError:  # Local scaffold checks can run before project dependencies.
    np = None

if np is not None:
    from gaussian.hil_capture import (
        evenly_spaced_indexes,
        parse_episode_spec,
        relative_data_path,
        relative_video_path,
        sampled_row_indexes,
    )
    from gaussian.hand_eye_alignment import solve_hand_eye_rotation
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        from gaussian.front_alignment import match_centers
        from gaussian.fixed_workspace import (
            active_image_crop,
            calibrate_front_camera,
            detect_target_quads,
        )
        from gaussian.planar_workspace import detect_yellow_quad, order_quad, track_quad
    from sim.genesis_so101.replay_hil import load_trajectory
else:
    cv2 = None


@unittest.skipIf(np is None, "numpy is not installed")
class HilReal2SimTests(unittest.TestCase):
    def test_episode_spec_supports_ranges_and_deduplicates(self):
        self.assertEqual(parse_episode_spec("4,1-3,3"), (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            parse_episode_spec("3-1")

    def test_sampling_is_timestamp_aligned(self):
        timestamps = [index / 30.0 for index in range(61)]
        self.assertEqual(sampled_row_indexes(timestamps, 2.0), (0, 15, 30, 45, 60))
        self.assertEqual(evenly_spaced_indexes(11, 3), (0, 5, 10))

    def test_lerobot_v3_source_paths_are_derived_from_metadata(self):
        record = {
            "data/chunk_index": 2,
            "data/file_index": 7,
            "videos/observation.images.hand_cam/chunk_index": 3,
            "videos/observation.images.hand_cam/file_index": 11,
        }
        self.assertEqual(relative_data_path(record).as_posix(), "data/chunk-002/file-007.parquet")
        self.assertEqual(
            relative_video_path("hand_cam", record).as_posix(),
            "videos/observation.images.hand_cam/chunk-003/file-011.mp4",
        )

    def test_colmap_text_model_counts_registered_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cameras.txt").write_text("# header\n1 OPENCV 640 480 500 500 320 240 0 0 0 0\n")
            (root / "images.txt").write_text(
                "# header\n"
                "1 1 0 0 0 0 0 0 1 frame_000.jpg\n\n"
                "2 1 0 0 0 1 0 0 1 frame_001.jpg\n1.0 2.0 -1\n"
            )
            (root / "points3D.txt").write_text("# header\n1 0 0 0 255 255 255 0.1 1 0\n")
            self.assertEqual(
                parse_text_model(root),
                {"cameras": 1, "registered_images": 2, "points3D": 1},
            )

    def test_hil_trajectory_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.npz"
            np.savez_compressed(
                path,
                action=np.zeros((3, 12), dtype=np.float32),
                observation_state=np.ones((3, 12), dtype=np.float32),
                timestamp=np.asarray((0.0, 1.0 / 30.0, 2.0 / 30.0)),
                frame_index=np.arange(3),
            )
            loaded = load_trajectory(path, "observation_state")
            self.assertEqual(loaded["observation_state"].shape, (3, 12))

    def test_hand_eye_rotation_recovers_synthetic_extrinsic(self):
        rng = np.random.default_rng(7)
        axis = np.asarray((0.2, -0.4, 0.7), dtype=np.float64)
        axis /= np.linalg.norm(axis)
        angle = 0.6
        skew = np.asarray(
            ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
        )
        expected = np.eye(4)
        expected[:3, :3] = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        poses = []
        for _ in range(10):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, -1] *= -1
            pose = np.eye(4)
            pose[:3, :3] = q
            poses.append(pose)
        pairs = []
        for first, second in zip(poses, poses[1:]):
            arm = np.linalg.inv(first) @ second
            camera = np.linalg.inv(expected) @ arm @ expected
            pairs.append((arm, camera))
        actual, errors = solve_hand_eye_rotation(pairs)
        np.testing.assert_allclose(actual, expected[:3, :3], atol=1e-6)
        self.assertLess(float(np.max(errors)), 1e-5)


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class PlanarWorkspaceTests(unittest.TestCase):
    def test_front_alignment_matches_visible_subset(self):
        reference = np.asarray(((66.0, 197.0), (287.0, 212.0), (524.0, 203.0)))
        simulation = np.asarray(((62.0, 199.0), (292.0, 212.0)))
        matches = match_centers(reference, simulation)
        self.assertEqual([(a, b) for a, b, _ in matches], [(0, 0), (1, 1)])
        self.assertLess(max(error for _, _, error in matches), 6.0)

    def test_yellow_quadrilateral_is_detected_and_ordered(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        points = np.asarray(((60, 50), (260, 65), (245, 200), (70, 190)), dtype=np.int32)
        cv2.fillConvexPoly(image, points, (0, 200, 240))
        quad, quality = detect_yellow_quad(image, min_area=1000, min_solidity=0.8)
        self.assertIsNotNone(quad)
        self.assertGreater(quality["area_px2"], 20_000)
        self.assertLess(float(quad[0, 0]), float(quad[1, 0]))

    def test_corner_tracking_preserves_identity_through_rotation(self):
        previous = np.asarray(((10, 10), (20, 10), (20, 20), (10, 20)), dtype=np.float32)
        current = previous[[2, 3, 0, 1]] + 1.0
        np.testing.assert_allclose(track_quad(previous, current), previous + 1.0)

    def test_fixed_camera_crop_and_multiple_targets(self):
        image = np.zeros((240, 400, 3), dtype=np.uint8)
        image[30:210] = (100, 100, 100)
        for left in (40, 165, 290):
            cv2.rectangle(image, (left, 80), (left + 60, 140), (0, 200, 240), -1)
        self.assertEqual(active_image_crop(image), (0, 30, 400, 180))
        self.assertEqual(len(detect_target_quads(image, min_area=500)), 3)

    def test_fixed_camera_calibration_recovers_synthetic_view(self):
        image_size = (640, 480)
        intrinsic = np.asarray(((700.0, 0.0, 320.0), (0.0, 700.0, 240.0), (0.0, 0.0, 1.0)))
        centers = [[0.2, -0.26], [0.0, -0.26], [-0.2, -0.26]]
        targets = []
        camera_position = np.asarray((0.0, 0.10, 0.90), dtype=np.float64)
        camera_forward = np.asarray((0.0, -0.26, 0.4105)) - camera_position
        camera_forward /= np.linalg.norm(camera_forward)
        camera_right = np.cross(camera_forward, np.asarray((0.0, 0.0, 1.0)))
        camera_right /= np.linalg.norm(camera_right)
        camera_down = np.cross(camera_forward, camera_right)
        world_to_camera = np.stack((camera_right, camera_down, camera_forward))
        rotation = cv2.Rodrigues(world_to_camera)[0]
        translation = -world_to_camera @ camera_position
        for center_x, center_y in centers:
            half = 0.04
            points = np.asarray(
                (
                    (center_x + half, center_y - half, 0.4105),
                    (center_x - half, center_y - half, 0.4105),
                    (center_x - half, center_y + half, 0.4105),
                    (center_x + half, center_y + half, 0.4105),
                ),
                dtype=np.float32,
            )
            corners = cv2.projectPoints(points, rotation, translation, intrinsic, None)[0]
            targets.append({"corners_px": corners.reshape(4, 2).tolist()})
        result = calibrate_front_camera(
            targets,
            image_size=image_size,
            world_centers_xy_m=centers,
            target_side_m=0.08,
            table_surface_z_m=0.4105,
        )
        self.assertTrue(result["accepted"])
        self.assertLess(result["fit"]["p95_reprojection_error_px"], 0.05)


if __name__ == "__main__":
    unittest.main()
