import unittest

from sim.genesis_so101.live_protocol import LeaderActionGate, LeaderActionPacket
from sim.genesis_so101.synthetic_leader_state import synthetic_action


class SyntheticLeaderStateTests(unittest.TestCase):
    def test_action_is_bounded_and_exercises_both_sides(self):
        start = synthetic_action(0.0)
        quarter = synthetic_action(0.25)
        self.assertEqual(len(start), 12)
        self.assertNotEqual(start[:6], quarter[:6])
        self.assertNotEqual(start[6:], quarter[6:])
        for action in (start, quarter, synthetic_action(0.5)):
            self.assertTrue(all(abs(value) <= 180.0 for value in action[:5] + action[6:11]))
            self.assertTrue(0.0 <= action[5] <= 100.0)
            self.assertTrue(0.0 <= action[11] <= 100.0)

    def test_consecutive_thirty_hz_packets_pass_live_gate(self):
        gate = LeaderActionGate()
        for index in range(30):
            gate.accept(
                LeaderActionPacket(
                    sequence_id=index,
                    captured_monotonic_ns=1_000_000_000 + index * 33_333_333,
                    captured_unix_ns=2_000_000_000 + index * 33_333_333,
                    action=synthetic_action(index / 180.0),
                )
            )

    def test_invalid_phase_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "phase"):
            synthetic_action(-0.1)


if __name__ == "__main__":
    unittest.main()
