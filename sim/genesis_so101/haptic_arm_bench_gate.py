#!/usr/bin/env python3
"""Validate the first time-bounded five-joint haptic hardware bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

from .haptic_hardware import BENCH_MOTORS


def evaluate_arm_bench(
    publisher: dict[str, Any],
    sender: dict[str, Any],
    preflight: dict[str, Any],
    authorization: dict[str, Any],
    *,
    expected_side: str,
    expected_full_scale: float,
    expected_reaction_effort: float,
) -> dict[str, Any]:
    haptic = publisher.get("haptic_feedback") or {}
    selection = haptic.get("arm_selection") or {}
    health_by_motor = haptic.get("latest_health") or {}
    preflight_checks = preflight.get("checks") or {}
    preflight_selection = preflight.get("selection") or {}
    preflight_envelope = preflight.get("command_envelope") or {}
    intervention = preflight.get("intervention") or {}
    required_preflight_checks = (
        "dual_arm_action_finite",
        "all_selected_motors_present",
        "all_selected_motors_torque_disabled",
        "all_selected_motors_position_mode",
        "all_selected_positions_have_bidirectional_model_margin",
        "all_selected_currents_within_bound",
        "all_selected_temperatures_within_bound",
        "all_selected_voltages_within_bound",
        "all_selected_status_clear",
        "synthetic_arm_command_envelope_accepted",
    )
    checks = {
        "authorization_exact_preflight_stage": (
            authorization.get("schema_version")
            == "radeon_oneloop.haptic_stage_authorization.v1"
            and authorization.get("accepted") is True
            and authorization.get("target_stage")
            == "single_arm_readonly_preflight"
            and authorization.get("physical_output_commands") is False
            and all(
                re.fullmatch(
                    r"[0-9a-f]{64}", str(authorization.get(field, ""))
                )
                is not None
                for field in (
                    "receipt_sha256",
                    "receipt_hash_index_sha256",
                    "receipt_done_sha256",
                )
            )
        ),
        "same_run_preflight_schema_and_selection": (
            preflight.get("schema_version")
            == "radeon_oneloop.haptic_arm_readonly_preflight.v1"
            and preflight.get("stage") == "single_arm_readonly_preflight"
            and preflight_selection.get("side") == expected_side
            and preflight_selection.get("motors") == list(BENCH_MOTORS)
        ),
        "same_run_preflight_envelope": (
            math.isclose(
                float(preflight_envelope.get("simulated_effort_full_scale", math.nan)),
                expected_full_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(preflight_envelope.get("reaction_effort", math.nan)),
                expected_reaction_effort,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and int(preflight_envelope.get("max_torque_limit_raw", -1)) == 20
            and math.isclose(
                float(
                    preflight_envelope.get(
                        "max_position_offset_limit_deg", math.nan
                    )
                ),
                0.5,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "same_run_preflight_accepted": (
            preflight.get("accepted") is True
            and all(
                preflight_checks.get(name) is True
                for name in required_preflight_checks
            )
            and preflight.get("physical_output_commands") is False
            and int(preflight.get("serial_register_writes", -1)) == 0
            and int(preflight.get("torque_enable_commands", -1)) == 0
        ),
        "same_process_intervention_transition": (
            preflight.get("same_process_transition") is True
            and preflight.get("bus_access")
            == "same_process_read_only_intervention_transition"
            and intervention.get("schema_version")
            == "radeon_oneloop.haptic_intervention_gate.v1"
            and intervention.get("mode") == "same_process_stable_safe_pose"
            and intervention.get("side") == expected_side
            and intervention.get("motors") == list(BENCH_MOTORS)
            and intervention.get("trigger")
            == "operator_attestation_plus_stable_safe_pose"
            and intervention.get("candidate_ready") is True
            and float(intervention.get("hold_required_s", 0.0)) >= 0.1
            and float(intervention.get("stable_duration_s", 0.0))
            >= float(intervention.get("hold_required_s", math.inf))
            and float(intervention.get("max_span_limit_deg", math.inf)) <= 2.0
            and intervention.get("serial_connection_preserved_for_arm") is True
        ),
        "publisher_schema_selection_and_rate": (
            publisher.get("schema_version")
            == "radeon_oneloop.leader_publisher.v1"
            and haptic.get("mode") == "physical-single-arm"
            and selection.get("side") == expected_side
            and selection.get("motors") == list(BENCH_MOTORS)
            and int(selection.get("max_torque_limit_raw", -1)) == 20
            and math.isclose(
                float(selection.get("max_position_offset_deg", math.nan)),
                0.5,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(selection.get("simulated_effort_full_scale", math.nan)),
                expected_full_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(selection.get("max_output_duration_s", math.nan)),
                5.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and 20.0 <= float(publisher.get("effective_hz", 0.0)) <= 35.0
            and int(publisher.get("send_errors", -1)) == 0
        ),
        "sender_schema_selection_and_rate": (
            sender.get("schema_version")
            == "radeon_oneloop.haptic_arm_bench_sender.v1"
            and sender.get("side") == expected_side
            and sender.get("motors") == list(BENCH_MOTORS)
            and int(sender.get("simultaneous_selected_channels", -1)) == 5
            and math.isclose(
                float(sender.get("reaction_effort", math.nan)),
                expected_reaction_effort,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and 20.0 <= float(sender.get("effective_hz", 0.0)) <= 35.0
            and int(sender.get("packets_sent", 0)) >= 150
            and sender.get("physical_output_commands") is False
        ),
        "feedback_transport_and_output_exercised": (
            int(haptic.get("accepted", 0)) >= 120
            and int(haptic.get("rejected", -1)) == 0
            and haptic.get("output_armed_ever") is True
            and int(haptic.get("output_commands", 0)) >= 120
            and haptic.get("physical_output_commands") is True
            and publisher.get("physical_output_commands") is True
        ),
        "all_motor_health_valid": (
            set(health_by_motor) == set(BENCH_MOTORS)
            and all(
                values.get("status") == 0
                and abs(int(values.get("present_current_raw", 10_000))) <= 150
                and int(values.get("present_temperature_c", 10_000)) <= 45
                and 60 <= int(values.get("present_voltage_raw", -1)) <= 84
                for values in health_by_motor.values()
            )
            and int(haptic.get("peak_abs_current_raw", 10_000)) <= 150
            and int(haptic.get("peak_temperature_c", 10_000)) <= 45
        ),
        "verified_fail_zero_shutdown": (
            haptic.get("shutdown_error") is None
            and haptic.get("release_attempted") is True
            and haptic.get("release_verified") is True
            and haptic.get("restore_verified") is True
            and haptic.get("output_armed_at_shutdown") is False
        ),
    }
    return {
        "schema_version": "radeon_oneloop.haptic_arm_bench_gate.v1",
        "formal": False,
        "stage": "single_arm_physical",
        "selected_side": expected_side,
        "accepted": all(checks.values()),
        "checks": checks,
        "physical_output_commands": True,
        "operator_perception_gate": "pending_separate_attestation",
        "next_candidate_stage": "dual_arm_monitor_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher", type=Path, required=True)
    parser.add_argument("--sender", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--full-scale", type=float, required=True)
    parser.add_argument("--reaction-effort", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_arm_bench(
        json.loads(args.publisher.read_text(encoding="utf-8")),
        json.loads(args.sender.read_text(encoding="utf-8")),
        json.loads(args.preflight.read_text(encoding="utf-8")),
        json.loads(args.authorization.read_text(encoding="utf-8")),
        expected_side=args.side,
        expected_full_scale=args.full_scale,
        expected_reaction_effort=args.reaction_effort,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("single-arm physical evidence gate failed")


if __name__ == "__main__":
    main()
