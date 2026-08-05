#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the full-geometry asset directory}"
asset_root=$ONELOOP_OBSERVED_CORE_ROOT
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
so101_assets=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
run_root=${ONELOOP_AUTHORITATIVE_SYNTHETIC_RUN_ROOT:-/home/amd/radeon-oneloop-runs/seva_full_geometry_authoritative_synthetic}
duration_s=${ONELOOP_AUTHORITATIVE_SYNTHETIC_DURATION_S:-6}
render_hz=${ONELOOP_AUTHORITATIVE_SYNTHETIC_RENDER_HZ:-8}
leader_port=${ONELOOP_AUTHORITATIVE_SYNTHETIC_LEADER_PORT:-58281}
visual_port=${ONELOOP_AUTHORITATIVE_SYNTHETIC_VISUAL_PORT:-58283}
timeout_s=${ONELOOP_AUTHORITATIVE_SYNTHETIC_TIMEOUT_S:-300}
publisher_duration_s=${ONELOOP_AUTHORITATIVE_SYNTHETIC_PUBLISH_DURATION_S:-90}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_gaussian_authoritative_synthetic"
run_dir="$run_root/$run_id"
consumer_dir="$run_dir/consumer"
renderer_dir="$run_dir/renderer"

[[ -x "$python_bin" ]]
[[ -f "$asset_root/appearance_full_geometry_canonical.ply" ]]
[[ -f "$asset_root/cameras_full_geometry.json" ]]
[[ -f "$asset_root/provenance.json" ]]
[[ -f "$so101_assets/so101_new_calib.xml" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
[[ "$leader_port" != "$visual_port" ]]
mkdir -p "$consumer_dir" "$renderer_dir"

renderer_pid=
consumer_pid=
publisher_pid=
cleanup() {
  task_status=$?
  for pid in "$publisher_pid" "$consumer_pid" "$renderer_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ $task_status -ne 0 && ! -e "$run_dir/DONE" && ! -e "$run_dir/FAILED" ]]; then
    printf '{"status":"failed","exit_code":%d}\n' "$task_status" >"$run_dir/FAILED"
  fi
  exit "$task_status"
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
  'schema_version: radeon_oneloop.amd_gaussian_authoritative_synthetic_gate_run.v1' \
  'formal: false' \
  'architecture: synthetic_leader_to_authoritative_genesis_to_non_authoritative_gaussian' \
  'serial_or_usb_access: false' \
  'physical_output: false' \
  'authoritative_object_visualization: false' \
  'renderer_object_visualization: false' \
  'compositor: gaussian_self_depth' \
  "duration_s: $duration_s" \
  "requested_render_hz: $render_hz" \
  "publisher_duration_s: $publisher_duration_s" \
  "asset_ply_sha256: $(sha256sum "$asset_root/appearance_full_geometry_canonical.ply" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

timeout --signal=TERM --kill-after=15 "$timeout_s" \
  "$python_bin" -m sim.genesis_so101.gaussian_live_view \
  --so101-asset-root "$so101_assets" \
  --observed-core-root "$asset_root" \
  --vksplat-root "$vksplat_root" \
  --output "$renderer_dir" \
  --bind-host 127.0.0.1 \
  --port "$visual_port" \
  --duration-s "$duration_s" \
  --first-packet-timeout-s 120 \
  --render-hz "$render_hz" \
  --record-video \
  --full-geometry-candidate \
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
  sleep 0.25
done

timeout --signal=TERM --kill-after=15 "$timeout_s" \
  "$python_bin" -m sim.genesis_so101.live_teleop \
  --asset-root "$so101_assets" \
  --output "$consumer_dir" \
  --bind-host 127.0.0.1 \
  --port "$leader_port" \
  --duration-s "$duration_s" \
  --first-packet-timeout-s 120 \
  --watchdog-ms 250 \
  --render-hz 0 \
  --hide-object-visualization \
  --visual-state-host 127.0.0.1 \
  --visual-state-port "$visual_port" \
  --visual-state-hz 30 \
  >"$run_dir/consumer.log" 2>&1 &
consumer_pid=$!

sleep 0.5
"$python_bin" -m sim.genesis_so101.synthetic_leader_state \
  --host 127.0.0.1 \
  --port "$leader_port" \
  --duration-s "$publisher_duration_s" \
  --cycle-s "$duration_s" \
  --hz 30 \
  --output "$run_dir/publisher_metrics.json" \
  >"$run_dir/publisher.log" 2>&1 &
publisher_pid=$!

wait "$consumer_pid"
consumer_pid=
wait "$renderer_pid"
renderer_pid=
kill -TERM "$publisher_pid" 2>/dev/null || true
wait "$publisher_pid"
publisher_pid=

"$python_bin" - "$consumer_dir/metrics.json" "$renderer_dir/metrics.json" "$run_dir/publisher_metrics.json" "$run_dir/gate.json" <<'PY'
import json
import sys
from pathlib import Path

consumer_path, renderer_path, publisher_path, output_path = map(Path, sys.argv[1:])
consumer = json.loads(consumer_path.read_text())
renderer = json.loads(renderer_path.read_text())
publisher = json.loads(publisher_path.read_text())
consumer_appearance = consumer["appearance"]["diagnostics"]
renderer_appearance = renderer["appearance"]
checks = {
    "authoritative_control_hz_at_least_110": consumer["sim_hz_effective"] >= 110.0,
    "authoritative_watchdog_events_zero": consumer["watchdog"]["events"] == 0,
    "leader_packet_rejections_zero": consumer["packets"]["rejected"] == 0,
    "visual_state_send_errors_zero": consumer["visual_state_stream"]["send_errors"] == 0,
    "authoritative_legacy_visual_mesh_hidden": (
        consumer_appearance["object_visualization"] is False
    ),
    "authoritative_collision_proxy_loaded": (
        consumer_appearance["object_mesh_path"].endswith("_collision.obj")
    ),
    "renderer_accepted": renderer["accepted"] is True,
    "renderer_full_geometry_candidate": renderer["full_geometry_candidate"] is True,
    "renderer_legacy_visual_mesh_hidden": (
        renderer_appearance["object_visualization"] is False
    ),
    "renderer_collision_proxy_loaded": (
        renderer_appearance["object_mesh_path"].endswith("_collision.obj")
    ),
    "renderer_self_depth": renderer_appearance["compositor"] == "gaussian_self_depth",
    "renderer_failures_and_fallbacks_zero": (
        renderer_appearance["binding"]["failures"] == 0
        and renderer_appearance["fallback_frames"] == 0
    ),
    "synthetic_publisher_accepted": publisher["accepted"] is True,
    "no_serial_or_usb_access": publisher["serial_or_usb_access"] is False,
    "physical_output_false": (
        consumer["physical_output_commands"] is False
        and renderer["physical_output"] is False
        and publisher["physical_output"] is False
    ),
}
report = {
    "schema_version": "radeon_oneloop.gaussian_authoritative_synthetic_gate.v1",
    "formal": False,
    "accepted": all(checks.values()),
    "checks": checks,
    "control_effective_hz": consumer["sim_hz_effective"],
    "render": renderer["render"],
    "leader_packets": consumer["packets"],
    "visual_snapshots": renderer["snapshots"],
    "authoritative_appearance": consumer_appearance,
    "renderer_appearance": renderer_appearance,
    "serial_or_usb_access": False,
    "physical_output": False,
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["accepted"]:
    raise RuntimeError("authoritative synthetic Gaussian gate failed")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_nonformal_authoritative_synthetic_gate"}\n' >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD authoritative synthetic Gaussian gate passed: %s\n' "$run_dir"
