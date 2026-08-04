import copy
import unittest

from sim.genesis_so101.haptic_arm_bench_gate import evaluate_arm_bench
from sim.genesis_so101.haptic_hardware import BENCH_MOTORS


FULL_SCALE = 0.6727447137236594
REACTION = 0.1345489427447319


def evidence():
    authorization = {
        "schema_version": "radeon_oneloop.haptic_stage_authorization.v1",
        "accepted": True,
        "target_stage": "single_arm_readonly_preflight",
        "receipt_sha256": "1" * 64,
        "receipt_hash_index_sha256": "2" * 64,
        "receipt_done_sha256": "3" * 64,
        "physical_output_commands": False,
    }
    preflight = {
        "schema_version": "radeon_oneloop.haptic_arm_readonly_preflight.v1",
        "stage": "single_arm_readonly_preflight",
        "accepted": True,
        "selection": {"side": "left", "motors": list(BENCH_MOTORS)},
        "checks": {
            "dual_arm_action_finite": True,
            "all_selected_motors_present": True,
            "all_selected_motors_torque_disabled": True,
            "all_selected_motors_position_mode": True,
            "all_selected_positions_have_bidirectional_model_margin": True,
            "all_selected_currents_within_bound": True,
            "all_selected_temperatures_within_bound": True,
            "all_selected_voltages_within_bound": True,
            "all_selected_status_clear": True,
            "synthetic_arm_command_envelope_accepted": True,
        },
        "command_envelope": {
            "simulated_effort_full_scale": FULL_SCALE,
            "reaction_effort": REACTION,
            "max_torque_limit_raw": 20,
            "max_position_offset_limit_deg": 0.5,
        },
        "serial_register_writes": 0,
        "torque_enable_commands": 0,
        "physical_output_commands": False,
        "same_process_transition": True,
        "bus_access": "same_process_read_only_intervention_transition",
        "intervention": {
            "schema_version": "radeon_oneloop.haptic_intervention_gate.v1",
            "mode": "same_process_stable_safe_pose",
            "side": "left",
            "motors": list(BENCH_MOTORS),
            "trigger": "operator_attestation_plus_stable_safe_pose",
            "candidate_ready": True,
            "hold_required_s": 0.4,
            "stable_duration_s": 0.42,
            "max_span_limit_deg": 2.0,
            "serial_connection_preserved_for_arm": True,
        },
    }
    health = {
        motor: {
            "present_current_raw": 2,
            "present_temperature_c": 32,
            "present_voltage_raw": 73,
            "status": 0,
        }
        for motor in BENCH_MOTORS
    }
    publisher = {
        "schema_version": "radeon_oneloop.leader_publisher.v1",
        "effective_hz": 29.9,
        "send_errors": 0,
        "physical_output_commands": True,
        "haptic_feedback": {
            "mode": "physical-single-arm",
            "accepted": 150,
            "rejected": 0,
            "arm_selection": {
                "side": "left",
                "motors": list(BENCH_MOTORS),
                "max_torque_limit_raw": 20,
                "max_position_offset_deg": 0.5,
                "simulated_effort_full_scale": FULL_SCALE,
                "max_output_duration_s": 5.0,
            },
            "output_armed_ever": True,
            "output_commands": 145,
            "latest_health": health,
            "peak_abs_current_raw": 2,
            "peak_temperature_c": 32,
            "physical_output_commands": True,
            "shutdown_error": None,
            "release_attempted": True,
            "release_verified": True,
            "restore_verified": True,
            "output_armed_at_shutdown": False,
        },
    }
    sender = {
        "schema_version": "radeon_oneloop.haptic_arm_bench_sender.v1",
        "side": "left",
        "motors": list(BENCH_MOTORS),
        "simultaneous_selected_channels": 5,
        "reaction_effort": REACTION,
        "effective_hz": 30.0,
        "packets_sent": 165,
        "physical_output_commands": False,
    }
    return publisher, sender, preflight, authorization


class HapticArmBenchGateTests(unittest.TestCase):
    def evaluate(self, values):
        return evaluate_arm_bench(
            *values,
            expected_side="left",
            expected_full_scale=FULL_SCALE,
            expected_reaction_effort=REACTION,
        )

    def test_accepts_complete_five_joint_evidence(self):
        report = self.evaluate(evidence())
        self.assertTrue(report["accepted"])
        self.assertTrue(all(report["checks"].values()))

    def test_rejects_stale_or_wrong_stage_authorization(self):
        values = list(copy.deepcopy(evidence()))
        values[3]["target_stage"] = "single_arm_physical"
        report = self.evaluate(values)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["authorization_exact_preflight_stage"])

    def test_rejects_any_unhealthy_motor(self):
        values = list(copy.deepcopy(evidence()))
        values[0]["haptic_feedback"]["latest_health"]["wrist_roll"][
            "present_temperature_c"
        ] = 50
        report = self.evaluate(values)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["all_motor_health_valid"])

    def test_rejects_unverified_shutdown(self):
        values = list(copy.deepcopy(evidence()))
        values[0]["haptic_feedback"]["release_verified"] = False
        report = self.evaluate(values)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["verified_fail_zero_shutdown"])

    def test_rejects_two_process_or_unstable_transition(self):
        values = list(copy.deepcopy(evidence()))
        values[2]["intervention"]["candidate_ready"] = False
        report = self.evaluate(values)
        self.assertFalse(report["accepted"])
        self.assertFalse(
            report["checks"]["same_process_intervention_transition"]
        )


if __name__ == "__main__":
    unittest.main()
