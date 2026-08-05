#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the content-verified asset directory}"
observed_core=$ONELOOP_OBSERVED_CORE_ROOT
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
run_root=${ONELOOP_ORBIT_AUDIT_RUN_ROOT:-$repo_root/runs/gaussian_orbit_audit}
frames=${ONELOOP_ORBIT_FRAMES:-72}
width=${ONELOOP_ORBIT_WIDTH:-512}
height=${ONELOOP_ORBIT_HEIGHT:-512}
candidate_nonformal=${ONELOOP_ORBIT_CANDIDATE_NONFORMAL:-0}
completed_appearance=${ONELOOP_COMPLETED_APPEARANCE:-0}
full_geometry_candidate=${ONELOOP_FULL_GEOMETRY_CANDIDATE:-0}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_gaussian_orbit_audit"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"

[[ -x "$python_bin" ]]
if [[ "$full_geometry_candidate" == 1 ]]; then
  appearance_ply="$observed_core/appearance_full_geometry_canonical.ply"
  camera_file="$observed_core/cameras_full_geometry.json"
elif [[ "$completed_appearance" == 1 ]]; then
  appearance_ply="$observed_core/appearance_completed_canonical.ply"
  camera_file="$observed_core/cameras_completed.json"
else
  appearance_ply="$observed_core/appearance_observed_canonical.ply"
  camera_file="$observed_core/cameras_observed.json"
fi
[[ -f "$appearance_ply" ]]
[[ -f "$camera_file" ]]
[[ -f "$observed_core/provenance.json" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
[[ "$candidate_nonformal" == 0 || "$candidate_nonformal" == 1 ]]
[[ "$completed_appearance" == 0 || "$completed_appearance" == 1 ]]
[[ "$full_geometry_candidate" == 0 || "$full_geometry_candidate" == 1 ]]
if (( candidate_nonformal + completed_appearance + full_geometry_candidate > 1 )); then
  printf '%s\n' 'appearance asset modes are mutually exclusive' >&2
  exit 64
fi
mkdir -p "$run_dir"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    printf '{"status":"failed","exit_code":%d}\n' "$status" >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap mark_failure EXIT

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

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_gaussian_orbit_audit_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_visual_audit' \
  "frames: $frames" \
  "image_size_wh: [$width, $height]" \
  'physical_output: false' \
  'eligible_for_heldout_real_metrics: false' \
  "candidate_nonformal: $candidate_nonformal" \
  "completed_appearance: $completed_appearance" \
  "full_geometry_candidate: $full_geometry_candidate" \
  "ply_sha256: $(sha256sum "$appearance_ply" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

audit_args=(
  -m sim.genesis_so101.gaussian_orbit_audit
  --asset-root "$observed_core" \
  --vksplat-root "$vksplat_root" \
  --output "$artifact_dir" \
  --frames "$frames" \
  --width "$width" \
  --height "$height"
)
if [[ "$candidate_nonformal" == 1 ]]; then
  audit_args+=(--candidate-nonformal)
fi
if [[ "$completed_appearance" == 1 ]]; then
  audit_args+=(--completed-appearance)
fi
if [[ "$full_geometry_candidate" == 1 ]]; then
  audit_args+=(--full-geometry-candidate)
fi
timeout --signal=TERM --kill-after=15 600 \
  "$python_bin" "${audit_args[@]}" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

"$python_bin" - "$artifact_dir/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())
if metrics.get("accepted_numeric") is not True:
    raise RuntimeError("orbit numeric acceptance is false")
if metrics.get("visual_review_required") is not True:
    raise RuntimeError("orbit run must preserve the visual-review requirement")
if metrics.get("physical_output") is not False:
    raise RuntimeError("orbit audit unexpectedly declares physical output")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_numeric_visual_review_required"}\n' >"$run_dir/DONE"
trap - EXIT
printf 'AMD Gaussian orbit audit passed numeric gates: %s\n' "$run_dir"
