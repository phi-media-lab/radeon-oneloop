#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rocm_python=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
so101_assets=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the full-geometry asset directory}"
asset_root=$ONELOOP_OBSERVED_CORE_ROOT
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
run_root=${ONELOOP_SYNTHETIC_LIVE_RUN_ROOT:-/home/amd/radeon-oneloop-runs/seva_full_geometry_synthetic_live}
render_duration_s=${ONELOOP_SYNTHETIC_RENDER_DURATION_S:-6}
publisher_duration_s=${ONELOOP_SYNTHETIC_PUBLISH_DURATION_S:-8}
render_hz=${ONELOOP_SYNTHETIC_RENDER_HZ:-8}
visual_port=${ONELOOP_SYNTHETIC_VISUAL_PORT:-58183}
timeout_s=${ONELOOP_SYNTHETIC_TIMEOUT_S:-240}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_gaussian_synthetic_live"
run_dir="$run_root/$run_id"
renderer_dir="$run_dir/renderer"

[[ -x "$rocm_python" ]]
[[ -f "$so101_assets/so101_new_calib.xml" ]]
[[ -f "$asset_root/appearance_full_geometry_canonical.ply" ]]
[[ -f "$asset_root/cameras_full_geometry.json" ]]
[[ -f "$asset_root/provenance.json" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
[[ "$visual_port" =~ ^[0-9]+$ ]] && (( visual_port >= 1 && visual_port <= 65535 ))
mkdir -p "$renderer_dir"

renderer_pid=
publisher_pid=
cleanup() {
  status=$?
  for pid in "$publisher_pid" "$renderer_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" && ! -e "$run_dir/FAILED" ]]; then
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
  'schema_version: radeon_oneloop.amd_gaussian_synthetic_live_gate_run.v1' \
  'formal: false' \
  'input: loopback_synthetic_udp_only' \
  'serial_or_usb_access: false' \
  'physical_output: false' \
  'object_visualization: false' \
  'compositor: gaussian_self_depth' \
  'collision_proxy: dedicated_invisible_collision_obj' \
  "render_duration_s: $render_duration_s" \
  "publisher_duration_s: $publisher_duration_s" \
  "requested_render_hz: $render_hz" \
  "visual_port: $visual_port" \
  "ply_sha256: $(sha256sum "$asset_root/appearance_full_geometry_canonical.ply" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

timeout --signal=TERM --kill-after=15 "$timeout_s" \
  "$rocm_python" -m sim.genesis_so101.gaussian_live_view \
  --so101-asset-root "$so101_assets" \
  --observed-core-root "$asset_root" \
  --vksplat-root "$vksplat_root" \
  --output "$renderer_dir" \
  --bind-host 127.0.0.1 \
  --port "$visual_port" \
  --duration-s "$render_duration_s" \
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

"$rocm_python" -m sim.genesis_so101.synthetic_visual_state \
  --host 127.0.0.1 \
  --port "$visual_port" \
  --duration-s "$publisher_duration_s" \
  --hz 30 \
  --output "$run_dir/publisher_metrics.json" \
  >"$run_dir/publisher.log" 2>&1 &
publisher_pid=$!

wait "$renderer_pid"
renderer_pid=
wait "$publisher_pid"
publisher_pid=

"$rocm_python" - "$renderer_dir/metrics.json" "$run_dir/publisher_metrics.json" "$run_dir/gate.json" <<'PY'
import json
import sys
from pathlib import Path

renderer_path, publisher_path, output_path = map(Path, sys.argv[1:])
renderer = json.loads(renderer_path.read_text())
publisher = json.loads(publisher_path.read_text())
appearance = renderer["appearance"]
checks = {
    "renderer_accepted": renderer["accepted"] is True,
    "full_geometry_candidate": renderer["full_geometry_candidate"] is True,
    "old_obj_visualization_disabled": appearance["object_visualization"] is False,
    "gaussian_self_depth_compositor": appearance["compositor"] == "gaussian_self_depth",
    "fallback_frames_zero": appearance["fallback_frames"] == 0,
    "binding_failures_zero": appearance["binding"]["failures"] == 0,
    "binding_rendered_both_cameras": (
        appearance["binding"]["successes"] == 2 * renderer["render"]["frames"]
    ),
    "snapshot_rejections_zero": renderer["snapshots"]["rejected"] == 0,
    "synthetic_publisher_accepted": publisher["accepted"] is True,
    "synthetic_hardware_access_false": publisher["hardware_access"] is False,
    "physical_output_false": (
        renderer["physical_output"] is False and publisher["physical_output"] is False
    ),
}
report = {
    "schema_version": "radeon_oneloop.gaussian_synthetic_live_gate.v1",
    "formal": False,
    "accepted": all(checks.values()),
    "checks": checks,
    "render": renderer["render"],
    "snapshots": renderer["snapshots"],
    "appearance": appearance,
    "trajectory": publisher["trajectory"],
    "hardware_access": False,
    "physical_output": False,
    "proves": [
        "dynamic pose binding over a full yaw sweep",
        "Gaussian self-depth compositing without the legacy visual OBJ",
        "two-camera live rendering from loopback snapshots",
    ],
    "does_not_prove": [
        "physics or collision fidelity",
        "real leader-arm teleoperation",
        "held-out real-view reconstruction accuracy",
    ],
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["accepted"]:
    raise RuntimeError("synthetic live gate failed")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_nonformal_synthetic_live_gate"}\n' >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD Gaussian synthetic live gate passed: %s\n' "$run_dir"
