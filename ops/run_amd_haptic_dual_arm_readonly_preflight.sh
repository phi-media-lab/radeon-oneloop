#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  printf 'usage: %s LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID DUAL_MONITOR_RECEIPT_RUN_DIR\n' "$0" >&2
  exit 64
fi

left_port=$1
right_port=$2
left_id=$3
right_id=$4
receipt_run_dir=$5
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_LEROBOT_PYTHON:-/home/amd/.miniforge3/envs/vlash/bin/python}
run_root=${ONELOOP_HAPTIC_DUAL_PREFLIGHT_RUN_ROOT:-$repo_root/runs/haptic_dual_arm_readonly_preflight}
full_scale=${ONELOOP_HAPTIC_SIMULATED_EFFORT_FULL_SCALE:-0.6727447137236594}
reaction_effort=${ONELOOP_HAPTIC_BENCH_REACTION_EFFORT:-0.1345489427447319}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_haptic_dual_arm_readonly_preflight"
run_dir="$run_root/$run_id"

[[ "$left_port" != "$right_port" ]]
[[ -e "$left_port" ]]
[[ -e "$right_port" ]]
[[ -x "$python_bin" ]]
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

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    files=("$run_dir/manifest.yaml")
    for path in authorization.json metrics.json stdout.log stderr.log; do
      if [[ -f "$run_dir/$path" ]]; then
        files+=("$run_dir/$path")
      fi
    done
    sha256sum "${files[@]}" >"$run_dir/hashes.sha256"
    printf '{"status":"failed","exit_code":%d}\n' "$status" \
      >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONPATH="$repo_root/src:$repo_root"
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_haptic_dual_arm_readonly_preflight_run.v1' \
  'formal: false' \
  'stage: dual_arm_readonly_preflight' \
  "authorization_receipt_run_id: $(basename "$receipt_run_dir")" \
  'selected_motor_count: 10' \
  'grippers_selected: false' \
  'max_torque_limit_raw_candidate: 15' \
  'max_position_offset_deg_candidate: 0.4' \
  "simulated_effort_full_scale_candidate: $full_scale" \
  "reaction_effort_candidate: $reaction_effort" \
  'watchdog_ms_candidate: 100' \
  'requires_prior_single_arm_empirical_acceptance: true' \
  'serial_register_writes: 0' \
  'physical_output_commands: false' \
  >"$run_dir/manifest.yaml"

"$python_bin" -m sim.genesis_so101.haptic_stage_authorize \
  --receipt "$receipt_run_dir/receipt.json" \
  --hash-index "$receipt_run_dir/hashes.sha256" \
  --done "$receipt_run_dir/DONE" \
  --target-stage dual_arm_readonly_preflight \
  --output "$run_dir/authorization.json"

timeout --signal=TERM --kill-after=3 45 \
  "$python_bin" -m sim.genesis_so101.haptic_dual_arm_readonly_preflight \
  --left-port "$left_port" \
  --right-port "$right_port" \
  --left-id "$left_id" \
  --right-id "$right_id" \
  --simulated-effort-full-scale "$full_scale" \
  --reaction-effort "$reaction_effort" \
  --max-torque-limit-raw 15 \
  --max-position-offset-deg 0.4 \
  --output "$run_dir/metrics.json" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

sha256sum \
  "$run_dir/manifest.yaml" \
  "$run_dir/authorization.json" \
  "$run_dir/metrics.json" \
  "$run_dir/stdout.log" \
  "$run_dir/stderr.log" \
  >"$run_dir/hashes.sha256"
sha256sum -c "$run_dir/hashes.sha256" >/dev/null
printf '{"status":"done_dual_arm_read_only_no_physical_output"}\n' \
  >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'dual-arm read-only preflight passed: %s\n' "$run_dir"
