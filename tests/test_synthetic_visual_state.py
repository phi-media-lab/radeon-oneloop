import math
import unittest

from sim.genesis_so101.synthetic_visual_state import synthetic_packet


class SyntheticVisualStateTests(unittest.TestCase):
    def test_packet_is_protocol_valid(self):
        packet = synthetic_packet(7, 0.25)
        packet.validate()
        self.assertEqual(packet.sequence_id, 7)
        self.assertEqual(len(packet.joint_positions_rad), 12)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in packet.object_quaternion_wxyz)),
            1.0,
        )

    def test_sweep_changes_pose_without_leaving_workspace(self):
        start = synthetic_packet(0, 0.0)
        quarter = synthetic_packet(1, 0.25)
        half = synthetic_packet(2, 0.5)
        self.assertNotEqual(start.object_position_m, quarter.object_position_m)
        self.assertNotEqual(start.object_quaternion_wxyz, half.object_quaternion_wxyz)
        for packet in (start, quarter, half):
            self.assertTrue(0.46 <= packet.object_position_m[2] <= 0.48)

    def test_invalid_phase_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "phase"):
            synthetic_packet(0, 1.01)


if __name__ == "__main__":
    unittest.main()
