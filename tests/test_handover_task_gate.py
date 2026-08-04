import unittest

import numpy as np

from sim.genesis_so101.handover_task_gate import HandoverTaskTracker
from sim.genesis_so101.scene import contact_pair_force_total


class HandoverTaskGateTests(unittest.TestCase):
    def _sample(
        self,
        tracker: HandoverTaskTracker,
        start: int,
        count: int,
        position,
        force,
    ) -> int:
        for index in range(start, start + count):
            tracker.update(
                sample_index=index,
                object_position_m=position,
                contact_force_n=force,
            )
        return start + count

    def test_accepts_ordered_left_to_right_handover_at_target(self):
        tracker = HandoverTaskTracker((0.10, -0.26, 0.47), minimum_samples=12)
        index = self._sample(tracker, 0, 3, (0.10, -0.26, 0.47), (0.0, 0.0))
        index = self._sample(tracker, index, 3, (0.06, -0.26, 0.50), (1.0, 0.0))
        index = self._sample(tracker, index, 3, (0.00, -0.26, 0.52), (1.0, 1.0))
        self._sample(tracker, index, 3, (-0.10, -0.26, 0.49), (0.0, 1.0))
        result = tracker.summary(
            target_position_m=(-0.10, -0.26, 0.47), target_tolerance_m=0.08
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(
            [item["event"] for item in result["events"]],
            ["left_grasp", "dual_handover", "left_release_right_holds"],
        )

    def test_rejects_direct_motion_without_dual_contact(self):
        tracker = HandoverTaskTracker((0.10, -0.26, 0.47), minimum_samples=12)
        index = self._sample(tracker, 0, 3, (0.10, -0.26, 0.47), (0.0, 0.0))
        index = self._sample(tracker, index, 3, (0.05, -0.26, 0.50), (1.0, 0.0))
        self._sample(tracker, index, 6, (-0.10, -0.26, 0.49), (0.0, 1.0))
        result = tracker.summary(
            target_position_m=(-0.10, -0.26, 0.47), target_tolerance_m=0.08
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(
            result["checks"]["ordered_left_grasp_dual_handover_left_release"]
        )

    def test_rejects_drop_even_with_contact_sequence(self):
        tracker = HandoverTaskTracker((0.10, -0.26, 0.47), minimum_samples=12)
        index = self._sample(tracker, 0, 3, (0.10, -0.26, 0.47), (0.0, 0.0))
        index = self._sample(tracker, index, 3, (0.05, -0.26, 0.50), (1.0, 0.0))
        index = self._sample(tracker, index, 3, (0.00, -0.26, 0.42), (1.0, 1.0))
        self._sample(tracker, index, 3, (-0.10, -0.26, 0.47), (0.0, 1.0))
        result = tracker.summary(
            target_position_m=(-0.10, -0.26, 0.47), target_tolerance_m=0.08
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["object_not_dropped_below_table_envelope"])

    def test_rejects_recovered_sequence_after_persistent_contact_loss(self):
        tracker = HandoverTaskTracker((0.10, -0.26, 0.47), minimum_samples=18)
        index = self._sample(tracker, 0, 3, (0.10, -0.26, 0.47), (0.0, 0.0))
        index = self._sample(tracker, index, 3, (0.06, -0.26, 0.50), (1.0, 0.0))
        index = self._sample(tracker, index, 3, (0.04, -0.26, 0.50), (0.0, 0.0))
        index = self._sample(tracker, index, 3, (0.00, -0.26, 0.52), (1.0, 1.0))
        self._sample(tracker, index, 6, (-0.10, -0.26, 0.49), (0.0, 1.0))
        result = tracker.summary(
            target_position_m=(-0.10, -0.26, 0.47), target_tolerance_m=0.08
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["sequence_violations"][0]["label"], "none")

    def test_object_contact_force_excludes_table_and_other_arm(self):
        geom_a = np.asarray([10, 10, 10, 30])
        geom_b = np.asarray([20, 40, 31, 10])
        forces = np.asarray([[3.0, 4.0, 0.0], [9.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]])
        total = contact_pair_force_total(geom_a, geom_b, forces, (10, 20), (20, 30))
        self.assertEqual(total, 5.0)


if __name__ == "__main__":
    unittest.main()
