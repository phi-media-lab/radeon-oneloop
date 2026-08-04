#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_graffiti_mickey_preview"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"
mesh="$repo_root/sim/genesis_so101/assets_generated/miniso_disney_fun_crash_graffiti_mickey_v1_sim_visual.obj"

[[ -x "$python_bin" ]]
[[ -f "$mesh" ]]
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
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_graffiti_mickey_preview.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_visual_qa' \
  'backend: Genesis_amdgpu' \
  'object_asset: miniso_disney_fun_crash_graffiti_mickey_v1' \
  "mesh_sha256: $(sha256sum "$mesh" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

timeout --signal=TERM --kill-after=10 180 \
  "$python_bin" -m sim.genesis_so101.render_handover_asset \
  --mesh "$mesh" \
  --output "$artifact_dir" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
touch "$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD Graffiti Mickey preview passed: %s\n' "$run_dir"
