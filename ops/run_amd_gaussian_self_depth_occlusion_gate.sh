#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the full-geometry asset directory}"
asset_root=$ONELOOP_OBSERVED_CORE_ROOT
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
so101_assets=${ONELOOP_SO101_ASSET_ROOT:-$repo_root/sim/genesis_so101/assets/so101}
collision_mesh="$repo_root/sim/genesis_so101/assets_generated/miniso_disney_fun_crash_graffiti_mickey_v1_collision.obj"
legacy_visual_mesh="$repo_root/sim/genesis_so101/assets_generated/miniso_disney_fun_crash_graffiti_mickey_v1_sim_visual.obj"
run_root=${ONELOOP_SELF_DEPTH_OCCLUSION_RUN_ROOT:-/home/amd/radeon-oneloop-runs/seva_full_geometry_self_depth_occlusion}
timeout_s=${ONELOOP_SELF_DEPTH_OCCLUSION_TIMEOUT_S:-300}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_gaussian_self_depth_occlusion"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"

[[ -x "$python_bin" ]]
[[ -f "$asset_root/appearance_full_geometry_canonical.ply" ]]
[[ -f "$asset_root/cameras_full_geometry.json" ]]
[[ -f "$asset_root/provenance.json" ]]
[[ -f "$so101_assets/so101_new_calib.xml" ]]
[[ -f "$collision_mesh" ]]
[[ -f "$legacy_visual_mesh" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
mkdir -p "$run_dir"

cleanup() {
  status=$?
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
  'schema_version: radeon_oneloop.amd_gaussian_self_depth_occlusion_gate_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_runtime_integration' \
  'serial_or_usb_access: false' \
  'physical_output: false' \
  'object_visualization: false' \
  'compositor: gaussian_self_depth' \
  "asset_ply_sha256: $(sha256sum "$asset_root/appearance_full_geometry_canonical.ply" | awk '{print $1}')" \
  "collision_mesh_sha256: $(sha256sum "$collision_mesh" | awk '{print $1}')" \
  "legacy_visual_mesh_sha256_negative_control_only: $(sha256sum "$legacy_visual_mesh" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

timeout --signal=TERM --kill-after=15 "$timeout_s" \
  "$python_bin" -m sim.genesis_so101.gaussian_self_depth_occlusion_gate \
  --so101-asset-root "$so101_assets" \
  --asset-root "$asset_root" \
  --vksplat-root "$vksplat_root" \
  --output "$artifact_dir" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

"$python_bin" - "$artifact_dir/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
checks = {
    "gate_accepted": report["accepted"] is True,
    "old_obj_visualization_false": report["old_obj_visualization"] is False,
    "legacy_visual_mesh_not_loaded": report["legacy_visual_mesh"]["loaded"] is False,
    "dedicated_collision_proxy_loaded": report["collision_proxy"]["loaded"] is True,
    "collision_proxy_invisible": report["collision_proxy"]["visualization"] is False,
    "binding_failures_zero": report["binding"]["failures"] == 0,
    "physical_output_false": report["physical_output"] is False,
}
if not all(checks.values()):
    raise RuntimeError(f"self-depth collision-path checks failed: {checks}")
print(json.dumps({"accepted": True, "checks": checks}, indent=2, sort_keys=True))
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_nonformal_self_depth_occlusion_gate"}\n' >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD Gaussian self-depth occlusion gate passed: %s\n' "$run_dir"
