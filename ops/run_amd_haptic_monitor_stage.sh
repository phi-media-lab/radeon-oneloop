#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  printf 'usage: %s LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID SIDE RECEIPT_RUN_DIR\n' "$0" >&2
  exit 64
fi

left_port=$1
right_port=$2
left_id=$3
right_id=$4
side=$5
receipt_run_dir=$6
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stage=single_arm_monitor_only
duration_s=${ONELOOP_HAPTIC_MONITOR_DURATION_S:-30}
timeout_s=${ONELOOP_HAPTIC_MONITOR_TIMEOUT_S:-240}
leader_port=${ONELOOP_LIVE_PORT:-58081}
feedback_port=${ONELOOP_HAPTIC_PORT:-58082}
run_root=${ONELOOP_HAPTIC_MONITOR_RUN_ROOT:-$repo_root/runs/haptic_monitor}
rocm_python=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
lerobot_python=${ONELOOP_LEROBOT_PYTHON:-/home/amd/.miniforge3/envs/vlash/bin/python}
asset_root=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_single_arm_monitor"
run_dir="$run_root/$run_id"

[[ "$left_port" != "$right_port" ]]
[[ -e "$left_port" ]]
[[ -e "$right_port" ]]
[[ -x "$rocm_python" ]]
[[ -x "$lerobot_python" ]]
[[ "$side" == left || "$side" == right ]]
[[ -f "$receipt_run_dir/receipt.json" ]]
[[ -f "$receipt_run_dir/hashes.sha256" ]]
[[ -f "$receipt_run_dir/DONE" ]]
"$lerobot_python" - "$duration_s" <<'PY'
import math, sys
value = float(sys.argv[1])
if not math.isfinite(value) or not 20.0 <= value <= 60.0:
    raise SystemExit("monitor duration must be between 20 and 60 seconds")
PY
for device in "$left_port" "$right_port"; do
  if fuser "$device" >/dev/null 2>&1; then
    printf 'serial device is busy: %s\n' "$device" >&2
    exit 75
  fi
done
mkdir -p "$run_dir"

consumer_pid=
publisher_pid=
write_hashes() {
  local files=("$run_dir/manifest.yaml")
  local name
  for name in authorization.json consumer.log publisher.log consumer/READY.json \
    consumer/metrics.json \
    publisher_metrics.json consumer/live_front_cam.png consumer/live_hand_cam.png \
    gate.json; do
    if [[ -f "$run_dir/$name" ]]; then
      files+=("$run_dir/$name")
    fi
  done
  sha256sum "${files[@]}" >"$run_dir/hashes.sha256"
}
cleanup() {
  status=$?
  if [[ -n "$publisher_pid" ]]; then
    kill -TERM "$publisher_pid" 2>/dev/null || true
    wait "$publisher_pid" 2>/dev/null || true
  fi
  if [[ -n "$consumer_pid" ]]; then
    kill -TERM "$consumer_pid" 2>/dev/null || true
    wait "$consumer_pid" 2>/dev/null || true
  fi
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    write_hashes
    printf '{"status":"failed_single_arm_monitor","exit_code":%d}\n' \
      "$status" >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONPATH="$repo_root/src:$repo_root"
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_haptic_monitor_run.v1' \
  'formal: false' \
  "stage: $stage" \
  "selected_side: $side" \
  "duration_s: $duration_s" \
  "authorization_receipt_run_id: $(basename "$receipt_run_dir")" \
  'minimum_body_span_deg: 3.0' \
  'minimum_gripper_span_pct: 5.0' \
  'maximum_quiet_arm_span: 2.0' \
  'render_hz: 0' \
  'physical_leader_haptic_output: false' \
  'physical_follower_output: false' \
  >"$run_dir/manifest.yaml"

"$lerobot_python" -m sim.genesis_so101.haptic_stage_authorize \
  --receipt "$receipt_run_dir/receipt.json" \
  --hash-index "$receipt_run_dir/hashes.sha256" \
  --done "$receipt_run_dir/DONE" \
  --target-stage "$stage" \
  --output "$run_dir/authorization.json"

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export XDG_RUNTIME_DIR="$run_dir/xdg-runtime"
mkdir -p "$XDG_RUNTIME_DIR" "$run_dir/consumer"
chmod 700 "$XDG_RUNTIME_DIR"

ready_file="$run_dir/consumer/READY.json"
timeout --signal=TERM --kill-after=10 "$timeout_s" \
  "$rocm_python" -m sim.genesis_so101.live_teleop \
  --asset-root "$asset_root" \
  --output "$run_dir/consumer" \
  --bind-host 127.0.0.1 \
  --port "$leader_port" \
  --duration-s "$duration_s" \
  --first-packet-timeout-s 180 \
  --watchdog-ms 250 \
  --render-hz 0 \
  --feedback-host 127.0.0.1 \
  --feedback-port "$feedback_port" \
  --feedback-hz 30 \
  --ready-file "$ready_file" \
  --start-delay-s 5 \
  >"$run_dir/consumer.log" 2>&1 &
consumer_pid=$!

sleep 1
"$lerobot_python" -m sim.genesis_so101.leader_publisher \
  --left-port "$left_port" \
  --right-port "$right_port" \
  --left-id "$left_id" \
  --right-id "$right_id" \
  --destination-host 127.0.0.1 \
  --destination-port "$leader_port" \
  --feedback-bind-host 127.0.0.1 \
  --feedback-port "$feedback_port" \
  --feedback-source-host 127.0.0.1 \
  --haptic-output-mode monitor \
  --action-range-start-file "$ready_file" \
  --hz 30 \
  --duration-s 0 \
  --print-every 0 \
  --metrics-output "$run_dir/publisher_metrics.json" \
  >"$run_dir/publisher.log" 2>&1 &
publisher_pid=$!

ready_deadline=$((SECONDS + 180))
while [[ ! -f "$ready_file" ]]; do
  if ! kill -0 "$consumer_pid" 2>/dev/null; then
    printf '%s\n' 'Genesis consumer exited before the monitor stage became ready' >&2
    exit 70
  fi
  if ! kill -0 "$publisher_pid" 2>/dev/null; then
    printf '%s\n' 'leader publisher exited before the monitor stage became ready' >&2
    exit 70
  fi
  if (( SECONDS >= ready_deadline )); then
    printf '%s\n' 'monitor stage did not become ready within 180 seconds' >&2
    exit 70
  fi
  sleep 0.2
done
printf 'READY: within 5 seconds, move every %s arm joint by >=3 deg and its gripper by >=5%%; keep the other arm still\n' "$side"

wait "$consumer_pid"
consumer_pid=
kill -TERM "$publisher_pid" 2>/dev/null || true
wait "$publisher_pid"
publisher_pid=

"$lerobot_python" -m sim.genesis_so101.haptic_monitor_gate \
  --consumer "$run_dir/consumer/metrics.json" \
  --publisher "$run_dir/publisher_metrics.json" \
  --authorization "$run_dir/authorization.json" \
  --side "$side" \
  --output "$run_dir/gate.json"

write_hashes
sha256sum -c "$run_dir/hashes.sha256" >/dev/null
printf '{"status":"done_single_arm_monitor_machine_accepted"}\n' \
  >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'single-arm monitor gate passed: %s\n' "$run_dir"
