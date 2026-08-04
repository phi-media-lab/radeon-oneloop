#!/usr/bin/env python3
"""Gate a no-output, single-arm haptic monitoring exercise."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

from radeon_oneloop.contracts import ACTION_NAMES


STAGE = "single_arm_monitor_only"


def _finite_span(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != len(ACTION_NAMES):
        return None
    try:
        span = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(item) or item < 0.0 for item in span):
        return None
    return span


def evaluate_single_arm_monitor(
    consumer: dict[str, Any],
    publisher: dict[str, Any],
    authorization: dict[str, Any],
    *,
    expected_side: str,
    minimum_body_span_deg: float = 3.0,
    minimum_gripper_span_pct: float = 5.0,
    maximum_quiet_span: float = 2.0,
) -> dict[str, Any]:
    if expected_side not in ("left", "right"):
        raise ValueError("expected_side must be left or right")
    for name, value in (
        ("minimum_body_span_deg", minimum_body_span_deg),
        ("minimum_gripper_span_pct", minimum_gripper_span_pct),
        ("maximum_quiet_span", maximum_quiet_span),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    duration_s = float(consumer.get("duration_s", 0.0))
    minimum_transport_samples = max(int(duration_s * 20.0), 1)
    consumer_feedback = consumer.get("haptic_feedback") or {}
    publisher_feedback = publisher.get("haptic_feedback") or {}
    action_range = publisher.get("action_range") or {}
    span = _finite_span(action_range.get("span"))
    selected_start = 0 if expected_side == "left" else 6
    quiet_start = 6 if expected_side == "left" else 0
    selected_span = span[selected_start : selected_start + 6] if span else []
    quiet_span = span[quiet_start : quiet_start + 6] if span else []
    clamping = consumer.get("input_clamping") or {}
    tracking = consumer.get("tracking_error") or {}
    packets = consumer.get("packets") or {}
    watchdog = consumer.get("watchdog") or {}
    layout = consumer.get("scene_layout") or {}

    checks = {
        "authorization_schema_and_target": (
            authorization.get("schema_version")
            == "radeon_oneloop.haptic_stage_authorization.v1"
            and authorization.get("accepted") is True
            and authorization.get("target_stage") == STAGE
        ),
        "authorization_is_nonphysical": (
            authorization.get("physical_output_commands") is False
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
        "consumer_schema_and_duration": (
            consumer.get("schema_version")
            == "radeon_oneloop.genesis_live_teleop.v1"
            and duration_s >= 20.0
            and consumer.get("ready_file_emitted") is True
            and float(consumer.get("operator_start_delay_s", 0.0)) >= 5.0
        ),
        "parallel_side_by_side_layout": (
            layout.get("arrangement") == "side_by_side_parallel"
            and math.isclose(
                float(layout.get("base_separation_m", math.nan)),
                0.40,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and layout.get("shared_base_euler_deg") == [0.0, 0.0, 0.0]
        ),
        "control_rate_and_transport": (
            float(consumer.get("sim_hz_effective", 0.0)) >= 100.0
            and int(packets.get("accepted", 0)) >= minimum_transport_samples
            and int(packets.get("rejected", -1)) == 0
        ),
        "watchdog_clear": (
            int(watchdog.get("events", -1)) == 0
            and watchdog.get("active_at_end") is False
        ),
        "zero_input_clamping": (
            int(clamping.get("processed_packets_with_clamping", -1)) == 0
            and int(clamping.get("processed_values_clamped", -1)) == 0
            and math.isclose(
                float(clamping.get("max_abs_delta", math.nan)),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "virtual_tracking_error_bounded": (
            float(tracking.get("max_abs", math.inf)) <= 1.0
        ),
        "consumer_feedback_transport": (
            consumer_feedback.get("enabled") is True
            and int(consumer_feedback.get("packets_sent", 0))
            >= minimum_transport_samples
            and int(consumer_feedback.get("send_errors", -1)) == 0
        ),
        "publisher_schema_rate_and_transport": (
            publisher.get("schema_version")
            == "radeon_oneloop.leader_publisher.v1"
            and 20.0 <= float(publisher.get("effective_hz", 0.0)) <= 35.0
            and int(publisher.get("samples", 0)) >= minimum_transport_samples
            and int(publisher.get("send_errors", -1)) == 0
            and publisher_feedback.get("mode") == "monitor"
            and int(publisher_feedback.get("accepted", 0))
            >= minimum_transport_samples
            and int(publisher_feedback.get("rejected", -1)) == 0
        ),
        "selected_arm_exercised": (
            action_range.get("action_names") == list(ACTION_NAMES)
            and action_range.get("capture_start_gated") is True
            and action_range.get("capture_started") is True
            and int(action_range.get("samples", 0)) >= minimum_transport_samples
            and len(selected_span) == 6
            and all(value >= minimum_body_span_deg for value in selected_span[:5])
            and selected_span[5] >= minimum_gripper_span_pct
        ),
        "unselected_arm_quiet": (
            len(quiet_span) == 6
            and all(value <= maximum_quiet_span for value in quiet_span)
        ),
        "physical_output_absent": (
            consumer.get("physical_output_commands") is False
            and consumer_feedback.get("physical_output_commands") is False
            and publisher.get("physical_output_commands") is False
            and publisher_feedback.get("physical_output_commands") is False
            and publisher_feedback.get("output_armed_ever") is False
            and int(publisher_feedback.get("output_commands", -1)) == 0
        ),
    }
    accepted = all(checks.values())
    return {
        "schema_version": "radeon_oneloop.haptic_monitor_gate.v1",
        "formal": False,
        "stage": STAGE,
        "selected_side": expected_side,
        "accepted": accepted,
        "checks": checks,
        "thresholds": {
            "minimum_body_span_deg": minimum_body_span_deg,
            "minimum_gripper_span_pct": minimum_gripper_span_pct,
            "maximum_quiet_span": maximum_quiet_span,
            "minimum_transport_samples": minimum_transport_samples,
            "minimum_sim_hz": 100.0,
            "maximum_tracking_error": 1.0,
        },
        "observed": {
            "selected_span": selected_span,
            "quiet_span": quiet_span,
            "sim_hz_effective": consumer.get("sim_hz_effective"),
            "tracking_error_max_abs": tracking.get("max_abs"),
        },
        "authorization": authorization,
        "next_stage_requires_operator_receipt": True,
        "next_candidate_stage": "single_arm_physical",
        "physical_output_commands": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--publisher", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_single_arm_monitor(
        json.loads(args.consumer.read_text(encoding="utf-8")),
        json.loads(args.publisher.read_text(encoding="utf-8")),
        json.loads(args.authorization.read_text(encoding="utf-8")),
        expected_side=args.side,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        raise RuntimeError("single-arm monitor-only evidence gate failed")


if __name__ == "__main__":
    main()
