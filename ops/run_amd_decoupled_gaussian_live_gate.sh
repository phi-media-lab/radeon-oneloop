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
timeout_s=${ONELOOP_LIVE_TIMEOUT_S:-600}
render_hz=${ONELOOP_RENDER_HZ:-5}
leader_port=${ONELOOP_LIVE_PORT:-58081}
feedback_port=${ONELOOP_HAPTIC_PORT:-58082}
visual_port=${ONELOOP_VISUAL_STATE_PORT:-58083}
fault_exit_after_frames=${ONELOOP_FAULT_EXIT_AFTER_FRAMES:-0}
show_presenter=${ONELOOP_SHOW_PRESENTER:-0}
presenter_host=${ONELOOP_PRESENTER_HOST:-127.0.0.1}
presenter_port=${ONELOOP_PRESENTER_PORT:-58084}
presenter_jpeg_quality=${ONELOOP_PRESENTER_JPEG_QUALITY:-90}
show_genesis_viewer=${ONELOOP_SHOW_GENESIS_VIEWER:-0}
record_video=${ONELOOP_RECORD_VIDEO:-0}
candidate_nonformal=${ONELOOP_LIVE_CANDIDATE_NONFORMAL:-0}
generated_fill_enabled=${ONELOOP_GENERATED_FILL_ENABLED:-0}
completed_appearance=${ONELOOP_COMPLETED_APPEARANCE:-0}
full_geometry_candidate=${ONELOOP_FULL_GEOMETRY_CANDIDATE:-0}
final_task_recording=${ONELOOP_FINAL_TASK_RECORDING:-0}
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
[[ -d "$vksplat_root/vksplat/shader" ]]
[[ "$fault_exit_after_frames" =~ ^[0-9]+$ ]]
[[ "$show_presenter" == 0 || "$show_presenter" == 1 ]]
[[ "$show_genesis_viewer" == 0 || "$show_genesis_viewer" == 1 ]]
[[ "$record_video" == 0 || "$record_video" == 1 ]]
[[ "$candidate_nonformal" == 0 || "$candidate_nonformal" == 1 ]]
[[ "$generated_fill_enabled" == 0 || "$generated_fill_enabled" == 1 ]]
[[ "$completed_appearance" == 0 || "$completed_appearance" == 1 ]]
[[ "$full_geometry_candidate" == 0 || "$full_geometry_candidate" == 1 ]]
[[ "$final_task_recording" == 0 || "$final_task_recording" == 1 ]]
if (( candidate_nonformal + generated_fill_enabled + completed_appearance + full_geometry_candidate > 1 )); then
  printf '%s\n' 'appearance asset modes are mutually exclusive' >&2
  exit 64
fi
if [[ "$full_geometry_candidate" == 1 ]]; then
  appearance_ply="$observed_core/appearance_full_geometry_canonical.ply"
  camera_file="$observed_core/cameras_full_geometry.json"
  [[ -f "$observed_core/provenance.json" ]]
elif [[ "$completed_appearance" == 1 ]]; then
  appearance_ply="$observed_core/appearance_completed_canonical.ply"
  camera_file="$observed_core/cameras_completed.json"
  [[ -f "$observed_core/provenance.json" ]]
elif [[ "$generated_fill_enabled" == 1 ]]; then
  appearance_ply="$observed_core/appearance_fused_preview.ply"
  camera_file="$observed_core/cameras_observed.json"
  [[ -f "$observed_core/appearance_fused_preview.provenance.json" ]]
else
  appearance_ply="$observed_core/appearance_observed_canonical.ply"
  camera_file="$observed_core/cameras_observed.json"
  [[ -f "$observed_core/provenance.json" ]]
fi
[[ -f "$appearance_ply" ]]
[[ -f "$camera_file" ]]
[[ "$presenter_host" == 127.0.0.1 || "$presenter_host" == localhost || "$presenter_host" == ::1 ]]
[[ "$presenter_port" =~ ^[0-9]+$ ]] && (( presenter_port >= 1 && presenter_port <= 65535 ))
[[ "$presenter_jpeg_quality" =~ ^[0-9]+$ ]] && (( presenter_jpeg_quality >= 50 && presenter_jpeg_quality <= 100 ))
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
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" && ! -e "$run_dir/STOPPED" ]]; then
    if [[ $status -eq 130 || $status -eq 143 ]]; then
      (cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name STOPPED ! -name FAILED \
        -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
      printf '{"status":"stopped_by_operator","exit_code":%d}\n' "$status" >"$run_dir/STOPPED"
    else
      (cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name STOPPED ! -name FAILED \
        -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
      printf '{"status":"failed","exit_code":%d}\n' "$status" >"$run_dir/FAILED"
    fi
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
if [[ "$show_presenter" == 1 || "$show_genesis_viewer" == 1 ]]; then
  desktop_runtime=${ONELOOP_DESKTOP_RUNTIME_DIR:-/run/user/$(id -u)}
  [[ -d "$desktop_runtime" ]]
  export XDG_RUNTIME_DIR="$desktop_runtime"
  export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$desktop_runtime/bus}
else
  xdg_runtime="$run_dir/xdg-runtime"
  mkdir "$xdg_runtime"
  chmod 700 "$xdg_runtime"
  export XDG_RUNTIME_DIR="$xdg_runtime"
fi

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_decoupled_gaussian_live_gate_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_runtime_integration' \
  'architecture: authoritative_control_plus_non_authoritative_renderer_process' \
  "duration_s: $duration_s" \
  'requested_control_hz: 120' \
  "requested_render_hz: $render_hz" \
  "show_presenter: $show_presenter" \
  "presenter_url: http://$presenter_host:$presenter_port/" \
  "show_genesis_viewer: $show_genesis_viewer" \
  "record_video: $record_video" \
  "candidate_nonformal: $candidate_nonformal" \
  'leader_bus_mode: read_only' \
  'physical_output: false' \
  "generated_fill_enabled: $generated_fill_enabled" \
  "completed_appearance: $completed_appearance" \
  "full_geometry_candidate: $full_geometry_candidate" \
  "final_task_recording: $final_task_recording" \
  "fault_exit_after_frames: $fault_exit_after_frames" \
  "ply_sha256: $(sha256sum "$appearance_ply" | awk '{print $1}')" \
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
if [[ "$show_presenter" == 1 ]]; then
  renderer_args+=(
    --present-http-host "$presenter_host"
    --present-http-port "$presenter_port"
    --present-jpeg-quality "$presenter_jpeg_quality"
  )
fi
if [[ "$record_video" == 1 ]]; then
  renderer_args+=(--record-video)
fi
if [[ "$candidate_nonformal" == 1 ]]; then
  renderer_args+=(--candidate-nonformal)
fi
if [[ "$generated_fill_enabled" == 1 ]]; then
  renderer_args+=(--layered-preview)
fi
if [[ "$completed_appearance" == 1 ]]; then
  renderer_args+=(--completed-appearance)
fi
if [[ "$full_geometry_candidate" == 1 ]]; then
  renderer_args+=(--full-geometry-candidate)
fi
if (( fault_exit_after_frames > 0 )); then
  renderer_args+=(--fault-exit-after-frames "$fault_exit_after_frames")
fi
timeout --signal=TERM --kill-after=15 "$timeout_s" \
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

if [[ "$show_presenter" == 1 ]]; then
  presenter_url=$("$rocm_python" - "$renderer_dir/READY" <<'PY'
import json
import sys
from pathlib import Path

ready = json.loads(Path(sys.argv[1]).read_text())
print(ready["presenter"]["url"])
PY
)
  timeout 15 xdg-open "$presenter_url" >"$run_dir/presenter_open.log" 2>&1 || \
    printf 'warning: could not open presenter URL automatically: %s\n' "$presenter_url" >&2
fi

consumer_args=(
  -m sim.genesis_so101.live_teleop
  --asset-root "$so101_assets"
  --output "$consumer_dir"
  --bind-host 127.0.0.1
  --port "$leader_port"
  --duration-s "$duration_s"
  --first-packet-timeout-s 180
  --watchdog-ms 250
  --render-hz 0
  --hide-object-visualization
  --feedback-host 127.0.0.1
  --feedback-port "$feedback_port"
  --feedback-hz 30
  --visual-state-host 127.0.0.1
  --visual-state-port "$visual_port"
  --visual-state-hz 30
)
if [[ "$final_task_recording" == 1 ]]; then
  consumer_args+=(--evaluate-handover)
fi
if [[ "$show_genesis_viewer" == 1 ]]; then
  consumer_args+=(--show-viewer)
fi
timeout --signal=TERM --kill-after=15 "$timeout_s" \
  "$rocm_python" "${consumer_args[@]}" \
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
  "$fault_exit_after_frames" "$renderer_status" "$show_presenter" \
  "$candidate_nonformal" "$generated_fill_enabled" "$completed_appearance" \
  "$full_geometry_candidate" "$final_task_recording" <<'PY'
import json, sys
from pathlib import Path

consumer_path, renderer_path, publisher_path, output_path, fault_path = map(Path, sys.argv[1:6])
fault_frames = int(sys.argv[6])
renderer_status = int(sys.argv[7])
show_presenter = bool(int(sys.argv[8]))
candidate_nonformal = bool(int(sys.argv[9]))
generated_fill_enabled = bool(int(sys.argv[10]))
completed_appearance = bool(int(sys.argv[11]))
full_geometry_candidate = bool(int(sys.argv[12]))
final_task_recording = bool(int(sys.argv[13]))
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
    "authoritative_old_obj_visualization_disabled": (
        consumer["appearance"]["diagnostics"]["object_visualization"] is False
    ),
    "authoritative_dedicated_collision_proxy_loaded": (
        consumer["appearance"]["diagnostics"]["object_mesh_path"].endswith(
            "_collision.obj"
        )
    ),
}
if final_task_recording:
    action_range = publisher.get("action_range") or {}
    span = action_range.get("span") or []
    left_arm_motion = sum(float(value) >= 5.0 for value in span[:5])
    right_arm_motion = sum(float(value) >= 5.0 for value in span[6:11])
    checks.update({
        "handover_task_accepted": consumer["handover_task"]["accepted"] is True,
        "left_arm_motion_coverage": left_arm_motion >= 3,
        "right_arm_motion_coverage": right_arm_motion >= 3,
        "left_gripper_exercised": len(span) == 12 and float(span[5]) >= 10.0,
        "right_gripper_exercised": len(span) == 12 and float(span[11]) >= 10.0,
        "task_trace_physical_output_false": (
            consumer["handover_task"]["physical_output_commands"] is False
        ),
    })
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
        "renderer_old_obj_visualization_disabled": (
            renderer["appearance"]["object_visualization"] is False
        ),
        "renderer_dedicated_collision_proxy_loaded": (
            renderer["appearance"]["object_mesh_path"].endswith("_collision.obj")
        ),
        "renderer_asset_mode_matches_request": (
            renderer["candidate_nonformal"] is candidate_nonformal
        ),
        "renderer_layered_mode_matches_request": (
            renderer["layered_preview"] is generated_fill_enabled
        ),
        "renderer_completed_mode_matches_request": (
            renderer["completed_appearance"] is completed_appearance
        ),
        "renderer_full_geometry_mode_matches_request": (
            renderer["full_geometry_candidate"] is full_geometry_candidate
        ),
    })
    if show_presenter:
        checks.update({
            "presenter_enabled": renderer["presenter"]["enabled"] is True,
            "presenter_received_all_frames": (
                renderer["presenter"]["frames_published"] == renderer["render"]["frames"]
            ),
            "presenter_page_requested": renderer["presenter"]["requests"]["page"] >= 1,
            "presenter_frame_requested": renderer["presenter"]["requests"]["frame"] >= 1,
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
    "generated_fill_enabled": generated_fill_enabled,
    "completed_appearance": completed_appearance,
    "full_geometry_candidate": full_geometry_candidate,
    "final_task_recording": final_task_recording,
    "handover_task": consumer["handover_task"] if final_task_recording else None,
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
