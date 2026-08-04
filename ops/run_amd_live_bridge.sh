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
duration_s=${ONELOOP_LIVE_DURATION_S:-30}
timeout_s=${ONELOOP_LIVE_TIMEOUT_S:-600}
port=${ONELOOP_LIVE_PORT:-58081}
feedback_port=${ONELOOP_HAPTIC_PORT:-58082}
haptic_output_mode=${ONELOOP_HAPTIC_OUTPUT_MODE:-monitor}
haptic_bench_side=${ONELOOP_HAPTIC_BENCH_SIDE:-}
haptic_bench_motor=${ONELOOP_HAPTIC_BENCH_MOTOR:-}
haptic_max_torque_raw=${ONELOOP_HAPTIC_MAX_TORQUE_RAW:-30}
haptic_max_offset_deg=${ONELOOP_HAPTIC_MAX_OFFSET_DEG:-1.0}
haptic_simulated_effort_full_scale=${ONELOOP_HAPTIC_SIMULATED_EFFORT_FULL_SCALE:-3.35}
haptic_max_output_duration_s=${ONELOOP_HAPTIC_MAX_OUTPUT_DURATION_S:-10}
physical_estop_confirmed=${ONELOOP_PHYSICAL_ESTOP_CONFIRMED:-0}
render_hz=${ONELOOP_RENDER_HZ:-0}
show_viewer=${ONELOOP_SHOW_VIEWER:-0}
appearance_mode=${ONELOOP_APPEARANCE_MODE:-debug-mesh}
observed_core_root=${ONELOOP_OBSERVED_CORE_ROOT:-}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs}
rocm_python=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
lerobot_python=${ONELOOP_LEROBOT_PYTHON:-/home/amd/.miniforge3/envs/vlash/bin/python}
asset_root=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_live_bridge"
run_dir="$run_root/$run_id"

[[ "$left_port" != "$right_port" ]]
[[ -e "$left_port" ]]
[[ -e "$right_port" ]]
[[ -x "$rocm_python" ]]
[[ -x "$lerobot_python" ]]
[[ "$show_viewer" == 0 || "$show_viewer" == 1 ]]
[[ "$haptic_output_mode" == monitor || "$haptic_output_mode" == bench-single-joint ]]
[[ "$appearance_mode" == debug-mesh || "$appearance_mode" == vksplat ]]
if [[ "$appearance_mode" == vksplat ]]; then
  [[ -n "$observed_core_root" ]]
  [[ -f "$observed_core_root/appearance_observed_canonical.ply" ]]
  [[ -d "$vksplat_root/vksplat/shader" ]]
fi
physical_output_commands=false
if [[ "$haptic_output_mode" == bench-single-joint ]]; then
  [[ "$physical_estop_confirmed" == 1 ]]
  [[ "$haptic_bench_side" == left || "$haptic_bench_side" == right ]]
  [[ -n "$haptic_bench_motor" ]]
  physical_output_commands=true
fi
mkdir -p "$run_dir"

publisher_pid=
consumer_pid=
write_hashes() {
  sha256sum \
    "$run_dir/manifest.yaml" \
    "$run_dir/metrics.json" \
    "$run_dir/live_front_cam.png" \
    "$run_dir/live_hand_cam.png" \
    "$run_dir/consumer.log" \
    "$run_dir/publisher.log" \
    >"$run_dir/hashes.sha256"
}

mark_failure() {
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
    if [[ ( $status -eq 130 || $status -eq 143 ) && -e "$run_dir/metrics.json" ]]; then
      write_hashes
      touch "$run_dir/STOPPED"
    else
      touch "$run_dir/FAILED"
    fi
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

export PYTHONPATH="$repo_root/src:$repo_root"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
if [[ "$show_viewer" == 1 ]]; then
  export XDG_RUNTIME_DIR=${ONELOOP_DESKTOP_RUNTIME_DIR:-/run/user/$(id -u)}
else
  export XDG_RUNTIME_DIR="$run_dir/xdg-runtime"
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
fi

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_live_bridge_run.v1' \
  'formal: false' \
  'host_role: amd_apu_live_demo' \
  "duration_s: $duration_s" \
  "udp_port: $port" \
  "haptic_udp_port: $feedback_port" \
  "haptic_mode: $haptic_output_mode" \
  "haptic_simulated_effort_full_scale: $haptic_simulated_effort_full_scale" \
  "render_hz: $render_hz" \
  "show_viewer: $show_viewer" \
  "appearance_mode: $appearance_mode" \
  'generated_fill_enabled: false' \
  "physical_leader_haptic_output: $physical_output_commands" \
  'physical_follower_output: false' \
  >"$run_dir/manifest.yaml"

consumer_args=(
  --asset-root "$asset_root"
  --output "$run_dir"
  --bind-host 127.0.0.1
  --port "$port"
  --duration-s "$duration_s"
  --first-packet-timeout-s 180
  --watchdog-ms 250
  --render-hz "$render_hz"
  --feedback-host 127.0.0.1
  --feedback-port "$feedback_port"
  --feedback-hz 30
  --appearance-mode "$appearance_mode"
)
if [[ "$appearance_mode" == vksplat ]]; then
  consumer_args+=(
    --observed-core-root "$observed_core_root"
    --vksplat-root "$vksplat_root"
  )
fi
if [[ "$show_viewer" == 1 ]]; then
  consumer_args+=(--show-viewer)
fi

timeout --signal=TERM --kill-after=10 "$timeout_s" \
  "$rocm_python" -m sim.genesis_so101.live_teleop \
  "${consumer_args[@]}" \
  >"$run_dir/consumer.log" 2>&1 &
consumer_pid=$!

sleep 1
publisher_args=(
  --left-port "$left_port"
  --right-port "$right_port"
  --left-id "$left_id"
  --right-id "$right_id"
  --destination-host 127.0.0.1
  --destination-port "$port"
  --feedback-bind-host 127.0.0.1
  --feedback-port "$feedback_port"
  --feedback-source-host 127.0.0.1
  --hz 30
  --duration-s 0
  --print-every 300
)
if [[ "$haptic_output_mode" == bench-single-joint ]]; then
  publisher_args+=(
    --haptic-output-mode bench-single-joint
    --haptic-bench-side "$haptic_bench_side"
    --haptic-bench-motor "$haptic_bench_motor"
    --haptic-max-torque-limit-raw "$haptic_max_torque_raw"
    --haptic-max-position-offset-deg "$haptic_max_offset_deg"
    --haptic-simulated-effort-full-scale "$haptic_simulated_effort_full_scale"
    --haptic-max-output-duration-s "$haptic_max_output_duration_s"
    --physical-estop-confirmed
  )
fi
"$lerobot_python" -m sim.genesis_so101.leader_publisher \
  "${publisher_args[@]}" \
  >"$run_dir/publisher.log" 2>&1 &
publisher_pid=$!

while kill -0 "$consumer_pid" 2>/dev/null && kill -0 "$publisher_pid" 2>/dev/null; do
  sleep 0.5
done

if ! kill -0 "$publisher_pid" 2>/dev/null && kill -0 "$consumer_pid" 2>/dev/null; then
  wait "$publisher_pid"
fi

wait "$consumer_pid"
consumer_pid=
kill -TERM "$publisher_pid" 2>/dev/null || true
wait "$publisher_pid"
publisher_pid=

write_hashes
touch "$run_dir/DONE"
trap - EXIT INT TERM
printf 'live bridge passed: %s\n' "$run_dir"
