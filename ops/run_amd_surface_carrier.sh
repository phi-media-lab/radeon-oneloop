#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
: "${ONELOOP_M1_MANIFEST:?set ONELOOP_M1_MANIFEST to the reviewed private M1 manifest}"
m1_manifest=$ONELOOP_M1_MANIFEST
run_root=${ONELOOP_SURFACE_CARRIER_RUN_ROOT:-$repo_root/runs/surface_carrier}
fit_steps=${ONELOOP_SURFACE_CARRIER_FIT_STEPS:-60}
fit_resolution=${ONELOOP_SURFACE_CARRIER_FIT_RESOLUTION:-64}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_surface_carrier"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"

[[ -x "$python_bin" ]]
[[ -f "$m1_manifest" ]]
[[ "$fit_steps" =~ ^[1-9][0-9]*$ ]]
[[ "$fit_resolution" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$run_dir"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    printf '{"status":"failed","exit_code":%d}\n' "$status" >"$run_dir/FAILED"
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

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_surface_carrier_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_surface_carrier' \
  'backend: torch_rocm_differentiable_fit_plus_deterministic_mesh_render' \
  'physical_output: false' \
  'eligible_for_heldout_real_metrics: false' \
  'redistribution: false' \
  "fit_steps: $fit_steps" \
  "fit_resolution: $fit_resolution" \
  "m1_manifest_sha256: $(sha256sum "$m1_manifest" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

timeout --signal=TERM --kill-after=30 1200 \
  "$python_bin" -m gaussian.surface_carrier \
  --m1-manifest "$m1_manifest" \
  --output "$artifact_dir" \
  --device cuda \
  --fit-steps "$fit_steps" \
  --fit-resolution "$fit_resolution" \
  --host-role amd_apu_nonformal_surface_carrier \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

"$python_bin" - "$artifact_dir/manifest.json" "$artifact_dir/DONE" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
done = json.loads(Path(sys.argv[2]).read_text())
if manifest.get("schema_version") != "radeon_oneloop.surface_carrier.v1":
    raise RuntimeError("unexpected surface-carrier schema")
if manifest.get("formal") is not False or manifest.get("physical_output") is not False:
    raise RuntimeError("surface-carrier provenance boundary was weakened")
if manifest.get("accepted_numeric") is not True:
    raise RuntimeError("surface-carrier numeric gate did not pass")
if manifest.get("visual_review_required") is not True:
    raise RuntimeError("surface-carrier run must preserve visual review")
if done.get("status") != "complete_numeric_candidate_visual_review_required":
    raise RuntimeError("surface-carrier output is incomplete")
PY

(cd "$run_dir" && find . -type f ! -name run_hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >run_hashes.sha256)
printf '{"status":"done_numeric_candidate_visual_review_required"}\n' >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'AMD surface carrier passed numeric gates: %s\n' "$run_dir"
