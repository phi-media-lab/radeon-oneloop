#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s LEFT_PORT RIGHT_PORT LEFT_CALIBRATION_ID RIGHT_CALIBRATION_ID\n' "$0" >&2
  exit 64
fi

left_port=$1
right_port=$2
left_id=$3
right_id=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rocm_python=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
lerobot_python=${ONELOOP_LEROBOT_PYTHON:-/home/amd/.miniforge3/envs/vlash/bin/python}
so101_assets=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the content-verified asset directory}"
observed_core=$ONELOOP_OBSERVED_CORE_ROOT
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
run_root=${ONELOOP_LIVE_RUN_ROOT:-/home/amd/radeon-oneloop-runs/live_gaussian_decoupled}
duration_s=${ONELOOP_LIVE_DURATION_S:-10}
render_hz=${ONELOOP_RENDER_HZ:-5}
leader_port=${ONELOOP_LIVE_PORT:-58081}
feedback_port=${ONELOOP_HAPTIC_PORT:-58082}
visual_port=${ONELOOP_VISUAL_STATE_PORT:-58083}
fault_exit_after_frames=${ONELOOP_FAULT_EXIT_AFTER_FRAMES:-0}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_decoupled_gaussian_live_gate"
run_dir="$run_root/$run_id"
consumer_dir="$run_dir/consumer"
renderer_dir="$run_dir/renderer"

[[ "$left_port" != "$right_port" ]]
[[ -e "$left_port" ]]
[[ -e "$right_port" ]]
[[ -x "$rocm_python" ]]
[[ -x "$lerobot_python" ]]
[[ -f "$so101_assets/so101_new_calib.xml" ]]
[[ -f "$observed_core/appearance_observed_canonical.ply" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
[[ "$fault_exit_after_frames" =~ ^[0-9]+$ ]]
mkdir -p "$consumer_dir" "$renderer_dir"

renderer_pid=
consumer_pid=
publisher_pid=
cleanup() {
  status=$?
  for pid in "$publisher_pid" "$consumer_pid" "$renderer_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$publisher_pid" "$consumer_pid" "$renderer_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    printf '{"status":"failed","exit_code":%d}\n' "$status" >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

export PYTHONPATH="$repo_root/src:$repo_root:$vksplat_root/build"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"
export XDG_RUNTIME_DIR="$xdg_runtime"

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_decoupled_gaussian_live_gate_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_runtime_integration' \
  'architecture: authoritative_control_plus_non_authoritative_renderer_process' \
  "duration_s: $duration_s" \
  'requested_control_hz: 120' \
  "requested_render_hz: $render_hz" \
  'leader_bus_mode: read_only' \
  'physical_output: false' \
  'generated_fill_enabled: false' \
  "fault_exit_after_frames: $fault_exit_after_frames" \
  "ply_sha256: $(sha256sum "$observed_core/appearance_observed_canonical.ply" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

renderer_args=(
  -m sim.genesis_so101.gaussian_live_view
  --so101-asset-root "$so101_assets" \
  --observed-core-root "$observed_core" \
  --vksplat-root "$vksplat_root" \
  --output "$renderer_dir" \
  --bind-host 127.0.0.1 \
  --port "$visual_port" \
  --duration-s "$duration_s" \
  --render-hz "$render_hz"
)
if (( fault_exit_after_frames > 0 )); then
  renderer_args+=(--fault-exit-after-frames "$fault_exit_after_frames")
fi
timeout --signal=TERM --kill-after=15 600 \
  "$rocm_python" "${renderer_args[@]}" \
  >"$run_dir/renderer.log" 2>&1 &
renderer_pid=$!

ready_deadline=$((SECONDS + 180))
while [[ ! -e "$renderer_dir/READY" ]]; do
  if ! kill -0 "$renderer_pid" 2>/dev/null; then
    wait "$renderer_pid"
  fi
  if (( SECONDS >= ready_deadline )); then
    printf '%s\n' 'renderer did not become ready within 180 seconds' >&2
    exit 70
  fi
  sleep 0.5
done

timeout --signal=TERM --kill-after=15 600 \
  "$rocm_python" -m sim.genesis_so101.live_teleop \
  --asset-root "$so101_assets" \
  --output "$consumer_dir" \
  --bind-host 127.0.0.1 \
  --port "$leader_port" \
  --duration-s "$duration_s" \
  --first-packet-timeout-s 180 \
  --watchdog-ms 250 \
  --render-hz 0 \
  --feedback-host 127.0.0.1 \
  --feedback-port "$feedback_port" \
  --feedback-hz 30 \
  --visual-state-host 127.0.0.1 \
  --visual-state-port "$visual_port" \
  --visual-state-hz 30 \
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
  --hz 30 \
  --duration-s 0 \
  --print-every 300 \
  >"$run_dir/publisher.log" 2>&1 &
publisher_pid=$!

wait "$consumer_pid"
consumer_pid=
kill -TERM "$publisher_pid" 2>/dev/null || true
wait "$publisher_pid"
publisher_pid=
set +e
wait "$renderer_pid"
renderer_status=$?
set -e
renderer_pid=
if (( fault_exit_after_frames == 0 && renderer_status != 0 )); then
  printf 'renderer exited unexpectedly with status %d\n' "$renderer_status" >&2
  exit "$renderer_status"
fi
if (( fault_exit_after_frames > 0 && renderer_status != 86 )); then
  printf 'fault-injected renderer exited with status %d, expected 86\n' "$renderer_status" >&2
  exit 71
fi

"$rocm_python" - "$consumer_dir/metrics.json" "$renderer_dir/metrics.json" \
  "$run_dir/publisher.log" "$run_dir/gate.json" "$renderer_dir/FAULT_INJECTED.json" \
  "$fault_exit_after_frames" "$renderer_status" <<'PY'
import json, sys
from pathlib import Path

consumer_path, renderer_path, publisher_path, output_path, fault_path = map(Path, sys.argv[1:6])
fault_frames = int(sys.argv[6])
renderer_status = int(sys.argv[7])
consumer = json.loads(consumer_path.read_text())
publisher_text = publisher_path.read_text()
decoder = json.JSONDecoder()
publisher = None
for index, char in enumerate(publisher_text):
    if char != "{":
        continue
    try:
        value, end = decoder.raw_decode(publisher_text[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and value.get("schema_version") == "radeon_oneloop.leader_publisher.v1":
        publisher = value
if publisher is None:
    raise RuntimeError("publisher summary is missing")

checks = {
    "control_effective_hz_at_least_110": consumer["sim_hz_effective"] >= 110.0,
    "control_watchdog_events_zero": consumer["watchdog"]["events"] == 0,
    "control_physical_output_false": consumer["physical_output_commands"] is False,
    "visual_stream_send_errors_zero": consumer["visual_state_stream"]["send_errors"] == 0,
    "publisher_physical_output_false": publisher["physical_output_commands"] is False,
}
if fault_frames:
    fault = json.loads(fault_path.read_text())
    checks.update({
        "renderer_hard_exit_observed": renderer_status == 86,
        "fault_marker_frame_count_matches": fault["frames_before_exit"] == fault_frames,
        "renderer_succeeded_before_fault": fault["binding"]["failures"] == 0,
    })
    render_effective_hz = None
    appearance_successes = fault["binding"]["successes"]
    appearance_failures = fault["binding"]["failures"]
    mode = "expected_renderer_hard_crash_isolation"
else:
    renderer = json.loads(renderer_path.read_text())
    checks.update({
        "renderer_accepted": renderer["accepted"] is True,
        "renderer_fallback_frames_zero": renderer["appearance"]["fallback_frames"] == 0,
    })
    render_effective_hz = renderer["render"]["effective_hz"]
    appearance_successes = renderer["appearance"]["binding"]["successes"]
    appearance_failures = renderer["appearance"]["binding"]["failures"]
    mode = "normal_decoupled_live_gate"
report = {
    "schema_version": "radeon_oneloop.decoupled_gaussian_live_gate.v1",
    "formal": False,
    "accepted": all(checks.values()),
    "checks": checks,
    "mode": mode,
    "control_effective_hz": consumer["sim_hz_effective"],
    "render_effective_hz": render_effective_hz,
    "appearance_successes": appearance_successes,
    "appearance_failures": appearance_failures,
    "physical_output": False,
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["accepted"]:
    raise RuntimeError("decoupled live gate checks failed")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_nonformal_decoupled_live_gate"}\n' >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD decoupled Gaussian live gate passed: %s\n' "$run_dir"
