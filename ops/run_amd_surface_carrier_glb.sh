#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs/surface_carrier_glb}
carrier_root=${ONELOOP_SURFACE_CARRIER_ROOT:?set ONELOOP_SURFACE_CARRIER_ROOT to an audited carrier artifact root}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_surface_carrier_glb"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"

[[ -x "$python_bin" ]]
[[ -f "$carrier_root/manifest.json" ]]
mkdir -p "$run_dir"

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

export PYTHONPATH="$repo_root/src:$repo_root"
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_surface_carrier_glb_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_portable_visual_conversion' \
  "source_manifest_sha256: $(sha256sum "$carrier_root/manifest.json" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

printf '%q ' \
  "$python_bin" -m gaussian.export_surface_carrier_glb \
  --surface-carrier-root "$carrier_root" --output "$artifact_dir" \
  >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

"$python_bin" -m gaussian.export_surface_carrier_glb \
  --surface-carrier-root "$carrier_root" \
  --output "$artifact_dir" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

(cd "$run_dir" && find . -type f ! -name run_hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >run_hashes.sha256)
touch "$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD surface-carrier GLB export passed: %s\n' "$run_dir"
