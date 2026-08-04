#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT}"
: "${ONELOOP_VISTA4D_SURFACE_CARRIER_ROOT:?set ONELOOP_VISTA4D_SURFACE_CARRIER_ROOT}"
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
run_root=${ONELOOP_MASK_ALIGNMENT_RUN_ROOT:-$repo_root/runs/vista4d_mask_alignment}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_vista4d_mask_alignment"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"

[[ -x "$python_bin" ]]
[[ -f "$ONELOOP_OBSERVED_CORE_ROOT/appearance_observed_canonical.ply" ]]
[[ -f "$ONELOOP_VISTA4D_SURFACE_CARRIER_ROOT/manifest.json" ]]
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
  'schema_version: radeon_oneloop.amd_vista4d_mask_alignment_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_mask_threshold_audit' \
  'physical_output: false' \
  "gaussian_ply_sha256: $(sha256sum "$ONELOOP_OBSERVED_CORE_ROOT/appearance_observed_canonical.ply" | awk '{print $1}')" \
  "carrier_manifest_sha256: $(sha256sum "$ONELOOP_VISTA4D_SURFACE_CARRIER_ROOT/manifest.json" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

timeout --signal=TERM --kill-after=15 300 \
  "$python_bin" -m gaussian.audit_vista4d_mask_alignment \
  --asset-root "$ONELOOP_OBSERVED_CORE_ROOT" \
  --vksplat-root "$vksplat_root" \
  --surface-carrier-root "$ONELOOP_VISTA4D_SURFACE_CARRIER_ROOT" \
  --output "$artifact_dir" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

"$python_bin" - "$artifact_dir/metrics.json" "$artifact_dir/DONE" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())
done = json.loads(Path(sys.argv[2]).read_text())
if metrics.get("schema_version") != "radeon_oneloop.vista4d_mask_alignment_audit.v1":
    raise RuntimeError("unexpected mask-alignment schema")
if metrics.get("formal") is not False or metrics.get("physical_output") is not False:
    raise RuntimeError("mask-alignment provenance boundary was weakened")
if done.get("status") != "complete_nonformal_threshold_selection":
    raise RuntimeError("mask-alignment audit is incomplete")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_nonformal_mask_threshold_selection"}\n' >"$run_dir/DONE"
trap - EXIT
printf 'AMD Vista4D mask alignment complete: %s\n' "$run_dir"
