import json
import unittest

from sim.genesis_so101.visual_state_protocol import (
    SCHEMA_VERSION,
    VisualStatePacket,
    VisualStateProtocolError,
    decode_visual_state,
    encode_visual_state,
)


class VisualStateProtocolTests(unittest.TestCase):
    def packet(self, sequence_id=4):
        return VisualStatePacket(
            sequence_id=sequence_id,
            captured_monotonic_ns=100,
            captured_unix_ns=200,
            joint_positions_rad=(0.0,) * 12,
            object_position_m=(0.1, -0.2, 0.47),
            object_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    def test_roundtrip(self):
        packet = self.packet()
        self.assertEqual(decode_visual_state(encode_visual_state(packet)), packet)

    def test_rejects_extra_field(self):
        document = json.loads(encode_visual_state(self.packet()))
        document["extra"] = True
        with self.assertRaisesRegex(VisualStateProtocolError, "fields"):
            decode_visual_state(json.dumps(document).encode())

    def test_rejects_non_normalized_quaternion(self):
        packet = self.packet()
        invalid = VisualStatePacket(
            packet.sequence_id,
            packet.captured_monotonic_ns,
            packet.captured_unix_ns,
            packet.joint_positions_rad,
            packet.object_position_m,
            (2.0, 0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(VisualStateProtocolError, "normalized"):
            encode_visual_state(invalid)

    def test_schema_is_frozen(self):
        self.assertEqual(SCHEMA_VERSION, "radeon_oneloop.visual_state.v1")


if __name__ == "__main__":
    unittest.main()
