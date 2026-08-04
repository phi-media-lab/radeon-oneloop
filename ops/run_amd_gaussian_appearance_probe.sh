#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the content-verified asset directory}"
asset_root=$ONELOOP_OBSERVED_CORE_ROOT
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
run_root=${ONELOOP_GAUSSIAN_RUN_ROOT:-/home/amd/radeon-oneloop-runs/gaussian_appearance}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_gaussian_appearance_probe"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"

[[ -x "$python_bin" ]]
[[ -f "$asset_root/appearance_observed_canonical.ply" ]]
[[ -f "$asset_root/cameras_observed.json" ]]
[[ -f "$asset_root/provenance.json" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
mkdir -p "$run_dir"

cleanup() {
  status=$?
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
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"
export XDG_RUNTIME_DIR="$xdg_runtime"

vulkaninfo --summary >"$run_dir/vulkaninfo_summary.txt" 2>"$run_dir/vulkaninfo_stderr.txt"
if ! grep -q 'vendorID.*0x1002' "$run_dir/vulkaninfo_summary.txt"; then
  printf '%s\n' 'no AMD Vulkan device found' >&2
  exit 69
fi
if ! grep -q 'driverName.*radv' "$run_dir/vulkaninfo_summary.txt"; then
  printf '%s\n' 'AMD Vulkan device is not using RADV' >&2
  exit 69
fi

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_gaussian_appearance_probe_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_runtime_integration' \
  'renderer: VkSplat_RADV' \
  'generated_fill_enabled: false' \
  'fallback: genesis_debug_mesh' \
  "ply_sha256: $(sha256sum "$asset_root/appearance_observed_canonical.ply" | awk '{print $1}')" \
  "cameras_sha256: $(sha256sum "$asset_root/cameras_observed.json" | awk '{print $1}')" \
  "provenance_sha256: $(sha256sum "$asset_root/provenance.json" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

timeout --signal=TERM --kill-after=10 300 \
  "$python_bin" -m sim.genesis_so101.gaussian_appearance_probe \
  --asset-root "$asset_root" \
  --vksplat-root "$vksplat_root" \
  --output "$artifact_dir" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_nonformal_capability_probe"}\n' >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD Gaussian appearance probe passed: %s\n' "$run_dir"
