import math
import unittest

from radeon_oneloop.contracts import (
    ACTION_NAMES,
    ActionLimits,
    ContractError,
    genesis_arm_to_lerobot,
    gripper_joint_to_percent,
    gripper_percent_to_joint,
    lerobot_arm_to_genesis,
    require_action_names,
)


class ContractTests(unittest.TestCase):
    def test_frozen_action_order(self):
        self.assertEqual(len(ACTION_NAMES), 12)
        require_action_names(ACTION_NAMES)
        with self.assertRaises(ContractError):
            require_action_names(tuple(reversed(ACTION_NAMES)))

    def test_gripper_round_trip(self):
        for percent in (0.0, 25.0, 50.0, 100.0):
            joint = gripper_percent_to_joint(percent, joint_min=-0.02, joint_max=0.04)
            self.assertAlmostEqual(
                gripper_joint_to_percent(joint, joint_min=-0.02, joint_max=0.04),
                percent,
            )

    def test_gripper_observation_tolerates_only_small_solver_overshoot(self):
        self.assertEqual(
            gripper_joint_to_percent(
                -0.0205, joint_min=-0.02, joint_max=0.04, tolerance=0.001
            ),
            0.0,
        )
        with self.assertRaises(ContractError):
            gripper_joint_to_percent(
                -0.022, joint_min=-0.02, joint_max=0.04, tolerance=0.001
            )

    def test_action_limits_and_delta(self):
        limits = ActionLimits((-1.0,) * 12, (1.0,) * 12, (0.1,) * 12)
        limits.validate((0.0,) * 12)
        limits.validate((0.05,) * 12, (0.0,) * 12)
        with self.assertRaises(ContractError):
            limits.validate((0.2,) + (0.0,) * 11, (0.0,) * 12)

    def test_lerobot_genesis_round_trip(self):
        physical = (-8.0, -90.0, 45.0, 75.0, 3.0, 62.5)
        recovered = genesis_arm_to_lerobot(lerobot_arm_to_genesis(physical))
        for left, right in zip(physical, recovered, strict=True):
            self.assertAlmostEqual(left, right)

    def test_runtime_can_select_a_bounded_solver_tolerance(self):
        genesis = list(lerobot_arm_to_genesis((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        genesis[-1] -= math.radians(0.6)
        with self.assertRaises(ContractError):
            genesis_arm_to_lerobot(genesis)
        recovered = genesis_arm_to_lerobot(
            genesis, gripper_tolerance_rad=math.radians(1.0)
        )
        self.assertEqual(recovered[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
