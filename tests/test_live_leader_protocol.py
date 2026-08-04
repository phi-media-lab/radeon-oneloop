import json
import unittest

from radeon_oneloop.contracts import ACTION_NAMES
from sim.genesis_so101.live_protocol import (
    LeaderActionGate,
    LeaderActionPacket,
    HapticFeedbackPacket,
    LiveProtocolError,
    clamp_action_to_model,
    decode_packet,
    decode_haptic_packet,
    encode_packet,
    encode_haptic_packet,
)


def packet(sequence_id: int, timestamp_ns: int, action=None) -> LeaderActionPacket:
    if action is None:
        action = (10.0, -50.0, 90.0, 40.0, 0.0, 25.0) * 2
    return LeaderActionPacket(
        sequence_id=sequence_id,
        captured_monotonic_ns=timestamp_ns,
        captured_unix_ns=1_800_000_000_000_000_000 + timestamp_ns,
        action=tuple(action),
    )


class LiveLeaderProtocolTests(unittest.TestCase):
    def test_round_trip_preserves_frozen_contract(self) -> None:
        original = packet(4, 1_000_000_000)
        decoded = decode_packet(encode_packet(original))
        self.assertEqual(decoded, original)
        self.assertEqual(tuple(decoded.as_dict()["action_names"]), ACTION_NAMES)

    def test_wrong_action_names_are_rejected(self) -> None:
        value = packet(0, 1_000_000_000).as_dict()
        value["action_names"][0] = "wrong.pos"
        with self.assertRaisesRegex(LiveProtocolError, "action_names"):
            decode_packet(json.dumps(value).encode())

    def test_non_finite_action_is_rejected(self) -> None:
        value = packet(0, 1_000_000_000).as_dict()
        value["action"][0] = float("nan")
        with self.assertRaisesRegex(LiveProtocolError, "non-finite"):
            decode_packet(json.dumps(value).encode())

    def test_gate_rejects_reordered_packet(self) -> None:
        gate = LeaderActionGate()
        gate.accept(packet(2, 1_000_000_000))
        with self.assertRaisesRegex(LiveProtocolError, "non-monotonic"):
            gate.accept(packet(2, 1_033_333_333))

    def test_gate_rejects_out_of_range_gripper(self) -> None:
        action = list(packet(0, 1_000_000_000).action)
        action[11] = 101.0
        with self.assertRaisesRegex(LiveProtocolError, "gripper range"):
            LeaderActionGate().accept(packet(0, 1_000_000_000, action))

    def test_gate_rejects_implausible_velocity_without_advancing_state(self) -> None:
        gate = LeaderActionGate(body_velocity_limit_deg_s=300.0)
        first = gate.accept(packet(0, 1_000_000_000))
        action = list(first.action)
        action[0] += 30.0
        with self.assertRaisesRegex(LiveProtocolError, "velocity"):
            gate.accept(packet(1, 1_033_333_333, action))
        self.assertEqual(gate.previous, first)

    def test_gate_accepts_normal_thirty_hz_motion(self) -> None:
        gate = LeaderActionGate(body_velocity_limit_deg_s=300.0)
        first = gate.accept(packet(0, 1_000_000_000))
        action = list(first.action)
        action[0] += 5.0
        second = gate.accept(packet(1, 1_033_333_333, action))
        self.assertEqual(gate.previous, second)

    def test_explicit_rebase_recovers_after_sender_gap(self) -> None:
        gate = LeaderActionGate(maximum_sender_dt_s=0.5)
        first = packet(5, 1_000_000_000)
        resumed = packet(20, 1_800_000_000)
        gate.accept(first)

        with self.assertRaisesRegex(LiveProtocolError, "sender gap"):
            gate.accept(resumed)
        self.assertEqual(gate.rebase(resumed), resumed)
        self.assertEqual(gate.previous, resumed)

    def test_rebase_keeps_sequence_order_fail_closed(self) -> None:
        gate = LeaderActionGate()
        current = packet(5, 1_000_000_000)
        gate.accept(current)

        with self.assertRaisesRegex(LiveProtocolError, "non-monotonic"):
            gate.rebase(packet(4, 2_000_000_000))
        self.assertEqual(gate.previous, current)

    def test_model_clamp_is_explicit_and_per_joint(self) -> None:
        action = list(packet(0, 1_000_000_000).action)
        action[1] = -102.9
        action[2] = 96.8
        action[7] = -102.9
        action[8] = 96.4

        clamped = clamp_action_to_model(tuple(action))

        self.assertEqual(clamped[1:3], (-100.0, 90.0))
        self.assertEqual(clamped[7:9], (-100.0, 90.0))
        self.assertEqual(clamped[0], action[0])

    def test_haptic_round_trip_preserves_effort_and_contact_force(self) -> None:
        packet_value = HapticFeedbackPacket(
            sequence_id=8,
            captured_monotonic_ns=2_000_000_000,
            captured_unix_ns=1_800_000_002_000_000_000,
            joint_reaction_effort=(0.1, -0.2, 0.0, 0.0, 0.0, 0.0) * 2,
            contact_force_n=(2.5, 1.25),
        )
        self.assertEqual(
            decode_haptic_packet(encode_haptic_packet(packet_value)), packet_value
        )

    def test_haptic_packet_rejects_non_finite_force(self) -> None:
        with self.assertRaisesRegex(LiveProtocolError, "contact_force_n"):
            HapticFeedbackPacket(
                sequence_id=0,
                captured_monotonic_ns=1,
                captured_unix_ns=1,
                joint_reaction_effort=(0.0,) * 12,
                contact_force_n=(float("nan"), 0.0),
            )


if __name__ == "__main__":
    unittest.main()
