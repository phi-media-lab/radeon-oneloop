#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  printf 'usage: %s LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID SIDE MONITOR_RECEIPT_RUN_DIR\n' "$0" >&2
  exit 64
fi

left_port=$1
right_port=$2
left_id=$3
right_id=$4
side=$5
receipt_run_dir=$6
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_LEROBOT_PYTHON:-/home/amd/.miniforge3/envs/vlash/bin/python}
feedback_port=${ONELOOP_HAPTIC_PORT:-58082}
run_root=${ONELOOP_HAPTIC_ARM_BENCH_RUN_ROOT:-$repo_root/runs/haptic_arm_bench}
estop_confirmed=${ONELOOP_PHYSICAL_ESTOP_CONFIRMED:-0}
workspace_clear=${ONELOOP_SELECTED_ARM_WORKSPACE_CLEAR_CONFIRMED:-0}
full_scale=${ONELOOP_HAPTIC_SIMULATED_EFFORT_FULL_SCALE:-0.6727447137236594}
reaction_effort=${ONELOOP_HAPTIC_BENCH_REACTION_EFFORT:-0.1345489427447319}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_haptic_arm_bench"
run_dir="$run_root/$run_id"

[[ "$left_port" != "$right_port" ]]
[[ -e "$left_port" ]]
[[ -e "$right_port" ]]
[[ -x "$python_bin" ]]
[[ "$side" == left || "$side" == right ]]
[[ "$estop_confirmed" == 1 ]]
[[ "$workspace_clear" == 1 ]]
[[ -f "$receipt_run_dir/receipt.json" ]]
[[ -f "$receipt_run_dir/hashes.sha256" ]]
[[ -f "$receipt_run_dir/DONE" ]]
for device in "$left_port" "$right_port"; do
  if fuser "$device" >/dev/null 2>&1; then
    printf 'serial device is busy: %s\n' "$device" >&2
    exit 75
  fi
done
mkdir -p "$run_dir"

publisher_pid=
sender_pid=
write_hashes() {
  local files=("$run_dir/manifest.yaml")
  local name
  for name in authorization.json preflight_stdout.log preflight_stderr.log \
    intervention_ready.json preflight_metrics.json publisher.log sender.log \
    publisher_metrics.json \
    sender_metrics.json gate.json; do
    if [[ -f "$run_dir/$name" ]]; then
      files+=("$run_dir/$name")
    fi
  done
  sha256sum "${files[@]}" >"$run_dir/hashes.sha256"
}
cleanup() {
  status=$?
  if [[ -n "$sender_pid" ]]; then
    kill -TERM "$sender_pid" 2>/dev/null || true
    wait "$sender_pid" 2>/dev/null || true
  fi
  if [[ -n "$publisher_pid" ]]; then
    kill -TERM "$publisher_pid" 2>/dev/null || true
    wait "$publisher_pid" 2>/dev/null || true
  fi
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    write_hashes
    printf '{"status":"failed_single_arm_physical","exit_code":%d}\n' \
      "$status" >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONPATH="$repo_root/src:$repo_root"
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_haptic_arm_bench_run.v1' \
  'formal: false' \
  'stage: single_arm_physical' \
  "side: $side" \
  "authorization_receipt_run_id: $(basename "$receipt_run_dir")" \
  'selected_motor_count: 5' \
  'gripper_selected: false' \
  'max_torque_limit_raw: 20' \
  'max_position_offset_deg: 0.5' \
  "simulated_effort_full_scale: $full_scale" \
  "synthetic_reaction_effort: $reaction_effort" \
  'max_output_duration_s: 5' \
  'watchdog_ms: 100' \
  'same_run_readonly_preflight_required: true' \
  'same_process_intervention_transition: true' \
  'intervention_stable_duration_s: 0.4' \
  'intervention_max_span_deg: 2.0' \
  'intervention_timeout_s: 90' \
  'operator_estop_attestation_received: true' \
  'operator_workspace_clear_attestation_received: true' \
  'physical_output_commands: true' \
  >"$run_dir/manifest.yaml"

"$python_bin" -m sim.genesis_so101.haptic_stage_authorize \
  --receipt "$receipt_run_dir/receipt.json" \
  --hash-index "$receipt_run_dir/hashes.sha256" \
  --done "$receipt_run_dir/DONE" \
  --target-stage single_arm_readonly_preflight \
  --output "$run_dir/authorization.json"

# The publisher keeps the same read-only serial connection while the operator
# places the light arm in a safe pose. After 0.4 s of stable margin it signals
# this wrapper to start feedback, re-reads all five motors, records the
# preflight boundary, and arms immediately without a gravity-sensitive process
# handoff.
timeout --signal=TERM --kill-after=3 105 \
  "$python_bin" -m sim.genesis_so101.leader_publisher \
  --left-port "$left_port" \
  --right-port "$right_port" \
  --left-id "$left_id" \
  --right-id "$right_id" \
  --destination-host 127.0.0.1 \
  --destination-port 58081 \
  --feedback-bind-host 127.0.0.1 \
  --feedback-port "$feedback_port" \
  --feedback-source-host 127.0.0.1 \
  --haptic-output-mode physical-single-arm \
  --haptic-bench-side "$side" \
  --haptic-max-torque-limit-raw 20 \
  --haptic-max-position-offset-deg 0.5 \
  --haptic-simulated-effort-full-scale "$full_scale" \
  --haptic-test-reaction-effort "$reaction_effort" \
  --haptic-max-output-duration-s 5 \
  --physical-estop-confirmed \
  --intervention-assisted-arm \
  --intervention-stable-duration-s 0.4 \
  --intervention-max-span-deg 2.0 \
  --intervention-timeout-s 90 \
  --intervention-ready-file "$run_dir/intervention_ready.json" \
  --intervention-preflight-output "$run_dir/preflight_metrics.json" \
  --hz 30 \
  --duration-s 0 \
  --print-every 0 \
  --metrics-output "$run_dir/publisher_metrics.json" \
  >"$run_dir/publisher.log" 2>&1 &
publisher_pid=$!

while [[ ! -f "$run_dir/intervention_ready.json" ]]; do
  if ! kill -0 "$publisher_pid" 2>/dev/null; then
    wait "$publisher_pid"
    publisher_pid=
    exit 1
  fi
  sleep 0.1
done

"$python_bin" -m sim.genesis_so101.haptic_arm_bench_sender \
  --host 127.0.0.1 \
  --port "$feedback_port" \
  --side "$side" \
  --duration-s 5.5 \
  --hz 30 \
  --contact-force-n 2.0 \
  --reaction-effort "$reaction_effort" \
  --metrics-output "$run_dir/sender_metrics.json" \
  >"$run_dir/sender.log" 2>&1 &
sender_pid=$!

wait "$publisher_pid"
publisher_pid=
wait "$sender_pid"
sender_pid=

"$python_bin" -m sim.genesis_so101.haptic_arm_bench_gate \
  --publisher "$run_dir/publisher_metrics.json" \
  --sender "$run_dir/sender_metrics.json" \
  --preflight "$run_dir/preflight_metrics.json" \
  --authorization "$run_dir/authorization.json" \
  --side "$side" \
  --full-scale "$full_scale" \
  --reaction-effort "$reaction_effort" \
  --output "$run_dir/gate.json"

write_hashes
sha256sum -c "$run_dir/hashes.sha256" >/dev/null
printf '{"status":"done_single_arm_physical_machine_accepted"}\n' \
  >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'single-arm physical gate passed: %s\n' "$run_dir"
