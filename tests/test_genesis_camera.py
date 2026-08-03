import unittest

try:
    import numpy as np
except ImportError:  # Local scaffold checks can run before project dependencies.
    np = None

if np is not None:
    from sim.genesis_so101.scene import relative_transform


@unittest.skipIf(np is None, "numpy is not installed")
class GenesisCameraTransformTests(unittest.TestCase):
    def test_relative_transform_reconstructs_world_pose(self) -> None:
        parent = np.eye(4)
        parent[:3, 3] = (2.0, -1.0, 0.5)
        child = np.eye(4)
        child[:3, :3] = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )
        child[:3, 3] = (-0.5, 3.0, 1.0)

        offset = relative_transform(parent, child)
        np.testing.assert_allclose(parent @ offset, child)

    def test_invalid_transform_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "4x4"):
            relative_transform(np.eye(3), np.eye(4))


if __name__ == "__main__":
    unittest.main()
