import unittest

from radeon_oneloop.contracts import ActionLimits
from radeon_oneloop.runtime_protocol import (
    ActionEnvelope,
    ObservationEnvelope,
    RuntimeState,
    SafetyController,
)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.now = 2_000_000_000
        limits = ActionLimits((-1.0,) * 12, (1.0,) * 12, (0.2,) * 12)
        self.controller = SafetyController(limits, clock_ns=lambda: self.now)
        self.controller.arm()

    def test_valid_chunk(self):
        obs = ObservationEnvelope(7, self.now - 1_000_000, (0.0,) * 12)
        act = ActionEnvelope(8, 7, self.now - 500_000, ((0.0,) * 12, (0.1,) * 12))
        self.assertEqual(len(self.controller.validate(obs, act)), 2)

    def test_stale_observation_latches_estop(self):
        obs = ObservationEnvelope(7, self.now - 1_000_000_000, (0.0,) * 12)
        act = ActionEnvelope(8, 7, self.now, ((0.0,) * 12,))
        with self.assertRaises(TimeoutError):
            self.controller.validate(obs, act)
        self.assertEqual(self.controller.state, RuntimeState.ESTOP)


if __name__ == "__main__":
    unittest.main()
