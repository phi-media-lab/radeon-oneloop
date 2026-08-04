#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
asset_root=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
run_root=${ONELOOP_HAPTIC_CALIBRATION_RUN_ROOT:-$repo_root/runs/haptic_contact_calibration}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_haptic_contact_calibration"
run_dir="$run_root/$run_id"

[[ -x "$python_bin" ]]
[[ -d "$asset_root" ]]
mkdir -p "$run_dir"

write_hashes() {
  sha256sum \
    "$run_dir/manifest.yaml" \
    "$run_dir/metrics.json" \
    "$run_dir/calibration.log" \
    >"$run_dir/hashes.sha256"
}

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    if [[ -f "$run_dir/metrics.json" && -f "$run_dir/calibration.log" ]]; then
      write_hashes
    fi
    touch "$run_dir/FAILED"
  fi
  exit "$status"
}
trap mark_failure EXIT

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

export PYTHONPATH="$repo_root/src:$repo_root"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_RUNTIME_DIR="$run_dir/xdg-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_haptic_contact_calibration_run.v1' \
  'formal: false' \
  'host_role: amd_apu_simulation_calibration' \
  'gpu_lock: /tmp/radeon-oneloop-gpu0.lock' \
  'physical_output_commands: false' \
  'serial_devices_opened: false' \
  'method: left_gripper_positive_x_face_mm_sweep' \
  >"$run_dir/manifest.yaml"

"$python_bin" -m sim.genesis_so101.haptic_contact_calibration \
  --asset-root "$asset_root" \
  --output "$run_dir" \
  >"$run_dir/calibration.log" 2>&1

write_hashes
sha256sum -c "$run_dir/hashes.sha256" >/dev/null
touch "$run_dir/DONE"
trap - EXIT
printf 'haptic contact calibration passed: %s\n' "$run_dir"
