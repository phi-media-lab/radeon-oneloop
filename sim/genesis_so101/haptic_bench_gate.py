#!/usr/bin/env python3
"""Validate a time-bounded single-joint haptic hardware evidence bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def evaluate_bench(
    publisher: dict[str, Any],
    sender: dict[str, Any],
    preflight: dict[str, Any],
    *,
    expected_side: str,
    expected_motor: str,
    expected_full_scale: float,
    expected_reaction_effort: float,
) -> dict[str, Any]:
    haptic = publisher.get("haptic_feedback", {})
    selection = haptic.get("bench_selection") or {}
    health = haptic.get("latest_health") or {}
    preflight_checks = preflight.get("checks") or {}
    preflight_selection = preflight.get("selection") or {}
    preflight_envelope = preflight.get("command_envelope") or {}
    intervention = preflight.get("intervention") or {}
    checks = {
        "preflight_schema": preflight.get("schema_version")
        == "radeon_oneloop.haptic_readonly_preflight.v1",
        "preflight_selection_and_calibration": (
            preflight_selection.get("side") == expected_side
            and preflight_selection.get("motor") == expected_motor
            and math.isclose(
                float(
                    preflight_envelope.get(
                        "simulated_effort_full_scale", math.nan
                    )
                ),
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
        ),
        "preflight_fail_closed_ready": (
            preflight.get("accepted") is True
            and preflight_checks.get("selected_motor_torque_disabled") is True
            and preflight_checks.get("selected_motor_position_mode") is True
            and preflight_checks.get(
                "selected_position_has_bidirectional_model_margin"
            )
            is True
            and preflight_checks.get("synthetic_command_envelope_accepted") is True
        ),
        "preflight_wrote_nothing": (
            preflight.get("physical_output_commands") is False
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
            and intervention.get("motors") == [expected_motor]
            and intervention.get("trigger")
            == "operator_attestation_plus_stable_safe_pose"
            and intervention.get("candidate_ready") is True
            and float(intervention.get("hold_required_s", 0.0)) >= 0.1
            and float(intervention.get("stable_duration_s", 0.0))
            >= float(intervention.get("hold_required_s", math.inf))
            and float(intervention.get("max_span_limit_deg", math.inf)) <= 2.0
            and intervention.get("serial_connection_preserved_for_arm") is True
        ),
        "publisher_schema": publisher.get("schema_version")
        == "radeon_oneloop.leader_publisher.v1",
        "sender_schema": sender.get("schema_version")
        == "radeon_oneloop.haptic_bench_sender.v1",
        "selection_matches": (
            selection.get("side") == expected_side
            and selection.get("motor") == expected_motor
            and math.isclose(
                float(selection.get("simulated_effort_full_scale", math.nan)),
                expected_full_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "sender_matches": (
            sender.get("side") == expected_side
            and sender.get("motor") == expected_motor
            and math.isclose(
                float(sender.get("reaction_effort", math.nan)),
                expected_reaction_effort,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "feedback_transport": (
            haptic.get("mode") == "bench-single-joint"
            and int(haptic.get("accepted", 0)) >= 250
            and int(haptic.get("rejected", -1)) == 0
        ),
        "output_exercised": (
            haptic.get("output_armed_ever") is True
            and int(haptic.get("output_commands", 0)) >= 250
            and publisher.get("physical_output_commands") is True
        ),
        "publisher_rate_and_send": (
            20.0 <= float(publisher.get("effective_hz", 0.0)) <= 35.0
            and int(publisher.get("send_errors", -1)) == 0
        ),
        "sender_rate_and_count": (
            20.0 <= float(sender.get("effective_hz", 0.0)) <= 35.0
            and int(sender.get("packets_sent", 0)) >= 300
            and sender.get("physical_output_commands") is False
        ),
        "health_status": (
            health.get("status") == 0
            and abs(int(health.get("present_current_raw", 10_000))) <= 150
            and int(health.get("present_temperature_c", 10_000)) <= 45
            and 60 <= int(health.get("present_voltage_raw", -1)) <= 84
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
        "schema_version": "radeon_oneloop.haptic_bench_gate.v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "physical_output_commands": True,
        "operator_perception_gate": "pending_separate_attestation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publisher", type=Path, required=True)
    parser.add_argument("--sender", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--motor", required=True)
    parser.add_argument("--full-scale", type=float, required=True)
    parser.add_argument("--reaction-effort", type=float, required=True)
    args = parser.parse_args()
    report = evaluate_bench(
        json.loads(args.publisher.read_text(encoding="utf-8")),
        json.loads(args.sender.read_text(encoding="utf-8")),
        json.loads(args.preflight.read_text(encoding="utf-8")),
        expected_side=args.side,
        expected_motor=args.motor,
        expected_full_scale=args.full_scale,
        expected_reaction_effort=args.reaction_effort,
    )
    payload = json.dumps(report, indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("single-joint haptic bench evidence gate failed")


if __name__ == "__main__":
    main()
