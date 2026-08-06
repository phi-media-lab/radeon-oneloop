#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs}
steps=${ONELOOP_TRELLIS2_SMOKE_STEPS:-20}
asset_root=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
asset_dir=$repo_root/sim/genesis_so101/assets_generated
asset_stem=graffiti_mickey_trellis2_real_front_seed12345
urdf=${ONELOOP_TRELLIS2_URDF:-$asset_dir/$asset_stem.urdf}
visual_mesh=$asset_dir/${asset_stem}_visual.obj
visual_texture=$asset_dir/${asset_stem}_texture.png
collision_mesh=${ONELOOP_TRELLIS2_COLLISION_MESH:-$asset_dir/miniso_disney_fun_crash_graffiti_mickey_v1_collision.obj}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_trellis2_mesh_smoke"
run_dir=$run_root/$run_id
artifact_dir=$run_dir/artifacts

[[ -x "$python_bin" ]]
[[ -f "$asset_root/so101_new_calib.xml" ]]
[[ -f "$urdf" ]]
[[ -f "$visual_mesh" ]]
[[ -f "$visual_texture" ]]
[[ -f "$collision_mesh" ]]
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
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export ONELOOP_RUN_DIR="$run_dir"
export XDG_RUNTIME_DIR="$run_dir/xdg-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_trellis2_mesh_smoke.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_real2sim_demo' \
  'backend: Genesis_amdgpu' \
  'appearance: TRELLIS2_textured_mesh' \
  'generated_visual_used_for_collision: false' \
  "steps: $steps" \
  "urdf_sha256: $(sha256sum "$urdf" | awk '{print $1}')" \
  "visual_mesh_sha256: $(sha256sum "$visual_mesh" | awk '{print $1}')" \
  "visual_texture_sha256: $(sha256sum "$visual_texture" | awk '{print $1}')" \
  "collision_mesh_sha256: $(sha256sum "$collision_mesh" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

printf '%q ' \
  "$python_bin" -m sim.genesis_so101.scripted_smoke \
  --asset-root "$asset_root" --output "$artifact_dir" --steps "$steps" \
  --object-urdf "$urdf" \
  >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

timeout --signal=TERM --kill-after=10 600 \
  "$python_bin" -m sim.genesis_so101.scripted_smoke \
  --asset-root "$asset_root" \
  --output "$artifact_dir" \
  --steps "$steps" \
  --object-urdf "$urdf" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
touch "$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD TRELLIS.2 mesh smoke passed: %s\n' "$run_dir"
