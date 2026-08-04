import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sim.genesis_so101.haptic_stage_receipt import (
    authorize_receipt_bundle,
    authorize_transition,
    build_single_arm_physical_receipt,
    build_single_arm_monitor_receipt,
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

    def test_sealed_receipt_bundle_authorizes_exact_monitor_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            gate, hashes, done = self.make_source(source)
            receipt = build_single_joint_receipt(
                source_run_id="20260804T000000Z_haptic",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                perception="useful_comfortable",
                leader_moves_freely_after_test=True,
            )
            receipt_run = root / "receipt-run"
            receipt_run.mkdir()
            receipt_path = receipt_run / "receipt.json"
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            receipt_hashes = receipt_run / "hashes.sha256"
            receipt_hashes.write_text(
                f"{_sha(receipt_path)}  {receipt_path}\n", encoding="utf-8"
            )
            receipt_done = receipt_run / "DONE"
            receipt_done.write_text(
                '{"status":"done_single_joint_stage_accepted"}\n',
                encoding="utf-8",
            )
            authorization = authorize_receipt_bundle(
                receipt_path=receipt_path,
                hash_index_path=receipt_hashes,
                done_path=receipt_done,
                target_stage="single_arm_monitor_only",
            )
            self.assertTrue(authorization["accepted"])
            self.assertFalse(authorization["physical_output_commands"])
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                authorize_receipt_bundle(
                    receipt_path=receipt_path,
                    hash_index_path=receipt_hashes,
                    done_path=receipt_done,
                    target_stage="single_arm_physical",
                )

    def test_tampered_sealed_receipt_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            gate, hashes, done = self.make_source(source)
            receipt = build_single_joint_receipt(
                source_run_id="20260804T000000Z_haptic",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                perception="useful_comfortable",
                leader_moves_freely_after_test=True,
            )
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            receipt_hashes = root / "hashes.sha256"
            receipt_hashes.write_text(
                f"{'0' * 64}  {receipt_path}\n", encoding="utf-8"
            )
            receipt_done = root / "DONE"
            receipt_done.write_text(
                '{"status":"done_single_joint_stage_accepted"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                authorize_receipt_bundle(
                    receipt_path=receipt_path,
                    hash_index_path=receipt_hashes,
                    done_path=receipt_done,
                    target_stage="single_arm_monitor_only",
                )

    def test_monitor_receipt_unlocks_preflight_not_physical_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.haptic_monitor_gate.v1",
                        "stage": "single_arm_monitor_only",
                        "selected_side": "left",
                        "accepted": True,
                        "physical_output_commands": False,
                        "next_stage_requires_operator_receipt": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            hashes = root / "hashes.sha256"
            hashes.write_text(f"{_sha(gate)}  {gate}\n", encoding="utf-8")
            done = root / "DONE"
            done.write_text(
                '{"status":"done_single_arm_monitor_machine_accepted"}\n',
                encoding="utf-8",
            )
            receipt = build_single_arm_monitor_receipt(
                source_run_id="20260804T000000Z_monitor",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                mapping_verdict="correct_same_side_same_direction",
                leader_moves_freely_after_monitor=True,
            )
            self.assertTrue(receipt["accepted"])
            authorize_transition(
                receipt, target_stage="single_arm_readonly_preflight"
            )
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                authorize_transition(receipt, target_stage="single_arm_physical")

    def test_incorrect_monitor_mapping_is_negative_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.haptic_monitor_gate.v1",
                        "stage": "single_arm_monitor_only",
                        "selected_side": "left",
                        "accepted": True,
                        "physical_output_commands": False,
                        "next_stage_requires_operator_receipt": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            hashes = root / "hashes.sha256"
            hashes.write_text(f"{_sha(gate)}  {gate}\n", encoding="utf-8")
            done = root / "DONE"
            done.write_text(
                '{"status":"done_single_arm_monitor_machine_accepted"}\n',
                encoding="utf-8",
            )
            receipt = build_single_arm_monitor_receipt(
                source_run_id="20260804T000000Z_monitor",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                mapping_verdict="wrong_side_or_direction",
                leader_moves_freely_after_monitor=True,
            )
            self.assertFalse(receipt["accepted"])
            self.assertIsNone(receipt["next_authorized_stage"])

    def test_arm_physical_receipt_unlocks_only_dual_arm_monitor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.haptic_arm_bench_gate.v1",
                        "stage": "single_arm_physical",
                        "selected_side": "left",
                        "accepted": True,
                        "physical_output_commands": True,
                        "operator_perception_gate": (
                            "pending_separate_attestation"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            hashes = root / "hashes.sha256"
            hashes.write_text(f"{_sha(gate)}  {gate}\n", encoding="utf-8")
            done = root / "DONE"
            done.write_text(
                '{"status":"done_single_arm_physical_machine_accepted"}\n',
                encoding="utf-8",
            )
            receipt = build_single_arm_physical_receipt(
                source_run_id="20260804T000000Z_arm",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                perception="useful_comfortable",
                no_cross_joint_instability=True,
                leader_moves_freely_after_test=True,
            )
            self.assertTrue(receipt["accepted"])
            authorize_transition(receipt, target_stage="dual_arm_monitor_only")
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                authorize_transition(receipt, target_stage="dual_arm_physical")

    def test_arm_instability_is_preserved_without_unlock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "schema_version": "radeon_oneloop.haptic_arm_bench_gate.v1",
                        "stage": "single_arm_physical",
                        "selected_side": "left",
                        "accepted": True,
                        "physical_output_commands": True,
                        "operator_perception_gate": (
                            "pending_separate_attestation"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            hashes = root / "hashes.sha256"
            hashes.write_text(f"{_sha(gate)}  {gate}\n", encoding="utf-8")
            done = root / "DONE"
            done.write_text(
                '{"status":"done_single_arm_physical_machine_accepted"}\n',
                encoding="utf-8",
            )
            receipt = build_single_arm_physical_receipt(
                source_run_id="20260804T000000Z_arm",
                gate_path=gate,
                source_hash_index_path=hashes,
                source_done_path=done,
                perception="useful_comfortable",
                no_cross_joint_instability=False,
                leader_moves_freely_after_test=True,
            )
            self.assertFalse(receipt["accepted"])
            self.assertIsNone(receipt["next_authorized_stage"])


if __name__ == "__main__":
    unittest.main()
