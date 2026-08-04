#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs}
steps=${ONELOOP_OBJECT_INTEGRATION_STEPS:-120}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_graffiti_mickey_integration"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"
config="$repo_root/configs/handover_object.json"
reference_manifest="$repo_root/data/graffiti_mickey_reference_sources.json"
mesh="$repo_root/sim/genesis_so101/assets_generated/miniso_disney_fun_crash_graffiti_mickey_v1_sim_visual.obj"
asset_root="$repo_root/sim/genesis_so101/assets/so101"

[[ -x "$python_bin" ]]
[[ -f "$config" ]]
[[ -f "$reference_manifest" ]]
[[ -f "$mesh" ]]
[[ -f "$asset_root/so101_new_calib.xml" ]]
mkdir -p "$artifact_dir"

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    touch "$run_dir/FAILED"
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

export PYTHONPATH="$repo_root/src:$repo_root"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ONELOOP_RUN_DIR="$run_dir"
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_graffiti_mickey_integration.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_physics' \
  'backend: Genesis_amdgpu' \
  'object_asset: miniso_disney_fun_crash_graffiti_mickey_v1' \
  "steps: $steps" \
  "config_sha256: $(sha256sum "$config" | awk '{print $1}')" \
  "reference_manifest_sha256: $(sha256sum "$reference_manifest" | awk '{print $1}')" \
  "mesh_sha256: $(sha256sum "$mesh" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

printf '%q ' \
  "$python_bin" -m sim.genesis_so101.scripted_smoke \
  --asset-root "$asset_root" --output "$artifact_dir" --steps "$steps" \
  >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

timeout --signal=TERM --kill-after=10 300 \
  "$python_bin" -m sim.genesis_so101.scripted_smoke \
  --asset-root "$asset_root" \
  --output "$artifact_dir" \
  --steps "$steps" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
touch "$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD Graffiti Mickey integration passed: %s\n' "$run_dir"
