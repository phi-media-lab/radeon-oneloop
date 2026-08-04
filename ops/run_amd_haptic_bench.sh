#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  printf 'usage: %s LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID SIDE MOTOR\n' "$0" >&2
  exit 64
fi

left_port=$1
right_port=$2
left_id=$3
right_id=$4
side=$5
motor=$6
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_LEROBOT_PYTHON:-/home/amd/.miniforge3/envs/vlash/bin/python}
feedback_port=${ONELOOP_HAPTIC_PORT:-58082}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs}
physical_estop_confirmed=${ONELOOP_PHYSICAL_ESTOP_CONFIRMED:-0}
workspace_clear=${ONELOOP_SELECTED_ARM_WORKSPACE_CLEAR_CONFIRMED:-0}
haptic_simulated_effort_full_scale=${ONELOOP_HAPTIC_SIMULATED_EFFORT_FULL_SCALE:-3.35}
haptic_bench_reaction_effort=${ONELOOP_HAPTIC_BENCH_REACTION_EFFORT:-3.35}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_haptic_bench"
run_dir="$run_root/$run_id"

[[ "$left_port" != "$right_port" ]]
[[ -e "$left_port" ]]
[[ -e "$right_port" ]]
[[ -x "$python_bin" ]]
[[ "$side" == left || "$side" == right ]]
[[ "$motor" == shoulder_pan || "$motor" == shoulder_lift || \
   "$motor" == elbow_flex || "$motor" == wrist_flex || \
   "$motor" == wrist_roll ]]
[[ "$physical_estop_confirmed" == 1 ]]
[[ "$workspace_clear" == 1 ]]
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
  for name in intervention_ready.json preflight_metrics.json publisher.log \
    sender.log publisher_metrics.json sender_metrics.json gate.json; do
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
    touch "$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONPATH="$repo_root/src:$repo_root"
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_haptic_bench_run.v1' \
  'formal: false' \
  'physical_output_commands: true' \
  "side: $side" \
  "motor: $motor" \
  'motor_count: 1' \
  'gripper_allowed: false' \
  'max_torque_limit_raw: 30' \
  'max_position_offset_deg: 1.0' \
  "simulated_effort_full_scale: $haptic_simulated_effort_full_scale" \
  "synthetic_reaction_effort: $haptic_bench_reaction_effort" \
  'max_output_duration_s: 10' \
  'watchdog_ms: 100' \
  'same_run_readonly_preflight_required: true' \
  'same_process_intervention_transition: true' \
  'intervention_stable_duration_s: 0.4' \
  'intervention_max_span_deg: 2.0' \
  'intervention_timeout_s: 90' \
  'operator_estop_attestation_received: true' \
  'operator_workspace_clear_attestation_received: true' \
  >"$run_dir/manifest.yaml"

# The same publisher connection waits torque-free for a stable safe position,
# records a zero-write register boundary, and arms only after feedback starts.
# This removes the gravity-sensitive preflight-to-publisher process handoff.
timeout --signal=TERM --kill-after=3 115 \
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
  --haptic-output-mode bench-single-joint \
  --haptic-bench-side "$side" \
  --haptic-bench-motor "$motor" \
  --haptic-max-torque-limit-raw 30 \
  --haptic-max-position-offset-deg 1.0 \
  --haptic-simulated-effort-full-scale "$haptic_simulated_effort_full_scale" \
  --haptic-test-reaction-effort "$haptic_bench_reaction_effort" \
  --haptic-max-output-duration-s 10 \
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

"$python_bin" -m sim.genesis_so101.haptic_bench_sender \
  --host 127.0.0.1 \
  --port "$feedback_port" \
  --side "$side" \
  --motor "$motor" \
  --duration-s 10.5 \
  --hz 30 \
  --contact-force-n 2.0 \
  --reaction-effort "$haptic_bench_reaction_effort" \
  --metrics-output "$run_dir/sender_metrics.json" \
  >"$run_dir/sender.log" 2>&1 &
sender_pid=$!

wait "$publisher_pid"
publisher_pid=
wait "$sender_pid"
sender_pid=

"$python_bin" -m sim.genesis_so101.haptic_bench_gate \
  --publisher "$run_dir/publisher_metrics.json" \
  --sender "$run_dir/sender_metrics.json" \
  --preflight "$run_dir/preflight_metrics.json" \
  --output "$run_dir/gate.json" \
  --side "$side" \
  --motor "$motor" \
  --full-scale "$haptic_simulated_effort_full_scale" \
  --reaction-effort "$haptic_bench_reaction_effort"

write_hashes
sha256sum -c "$run_dir/hashes.sha256" >/dev/null
touch "$run_dir/DONE"
trap - EXIT INT TERM
printf 'haptic bench passed: %s\n' "$run_dir"
