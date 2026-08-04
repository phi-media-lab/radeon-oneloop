import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sim.genesis_so101.haptic_stage_receipt import (
    authorize_transition,
    build_single_joint_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HapticStageReceiptTests(unittest.TestCase):
    def make_source(self, root: Path):
        gate = root / "gate.json"
        hashes = root / "hashes.sha256"
        done = root / "DONE"
        gate.write_text(
            json.dumps(
                {
                    "schema_version": "radeon_oneloop.haptic_bench_gate.v1",
                    "accepted": True,
                    "physical_output_commands": True,
                    "operator_perception_gate": "pending_separate_attestation",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        done.write_text("", encoding="utf-8")
        hashes.write_text(f"{_sha(gate)}  /private/run/gate.json\n", encoding="utf-8")
        return gate, hashes, done

    def test_accepted_operator_receipt_unlocks_only_single_arm_monitor(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate, hashes, done = self.make_source(Path(temporary))
            receipt = build_single_joint_receipt(
                source_run_id="20260804T000000Z_haptic",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                perception="useful_comfortable",
                leader_moves_freely_after_test=True,
            )
            self.assertTrue(receipt["accepted"])
            authorize_transition(receipt, target_stage="single_arm_monitor_only")
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                authorize_transition(receipt, target_stage="single_arm_physical")

    def test_weak_feedback_is_preserved_but_does_not_unlock_next_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate, hashes, done = self.make_source(Path(temporary))
            receipt = build_single_joint_receipt(
                source_run_id="20260804T000000Z_haptic",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                perception="too_weak",
                leader_moves_freely_after_test=True,
            )
            self.assertFalse(receipt["accepted"])
            with self.assertRaisesRegex(ValueError, "not accepted"):
                authorize_transition(receipt, target_stage="single_arm_monitor_only")

    def test_tampered_machine_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate, hashes, done = self.make_source(Path(temporary))
            gate.write_text("{}\n", encoding="utf-8")
            receipt = build_single_joint_receipt(
                source_run_id="20260804T000000Z_haptic",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                perception="useful_comfortable",
                leader_moves_freely_after_test=True,
            )
            self.assertFalse(receipt["accepted"])
            self.assertFalse(receipt["checks"]["gate_bound_by_source_hash_index"])


if __name__ == "__main__":
    unittest.main()
