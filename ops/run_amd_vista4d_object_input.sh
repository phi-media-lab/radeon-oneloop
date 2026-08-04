#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the content-verified asset directory}"
observed_core=$ONELOOP_OBSERVED_CORE_ROOT
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
run_root=${ONELOOP_VISTA4D_INPUT_RUN_ROOT:-$repo_root/runs/vista4d_object_input}
candidate_nonformal=${ONELOOP_VISTA4D_CANDIDATE_NONFORMAL:-0}
real_view_root=${ONELOOP_VISTA4D_REAL_VIEW_ROOT:-}
surface_carrier_root=${ONELOOP_VISTA4D_SURFACE_CARRIER_ROOT:-}
allow_rejected_surface_carrier=${ONELOOP_ALLOW_REJECTED_SURFACE_CARRIER_ABLATION:-0}
alpha_threshold=${ONELOOP_VISTA4D_ALPHA_THRESHOLD:-0.001}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_vista4d_object_input"
run_dir="$run_root/$run_id"
artifact_dir="$run_dir/artifacts"

[[ -x "$python_bin" ]]
[[ -f "$observed_core/appearance_observed_canonical.ply" ]]
[[ -f "$observed_core/cameras_observed.json" ]]
[[ -f "$observed_core/provenance.json" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
[[ "$candidate_nonformal" == 0 || "$candidate_nonformal" == 1 ]]
[[ "$allow_rejected_surface_carrier" == 0 || "$allow_rejected_surface_carrier" == 1 ]]
if [[ -n "$real_view_root" && -n "$surface_carrier_root" ]]; then
  printf '%s\n' 'real-view keyframes and surface-carrier source are mutually exclusive' >&2
  exit 64
fi
if [[ -n "$surface_carrier_root" && "$allow_rejected_surface_carrier" != 1 ]]; then
  printf '%s\n' 'surface carrier is rejected; set the explicit ablation flag only for failure reproduction' >&2
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
  'schema_version: radeon_oneloop.amd_vista4d_object_input_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_conditioning_render' \
  'frames: 49' \
  'image_size_wh: [672, 384]' \
  'physical_output: false' \
  'eligible_for_heldout_real_metrics: false' \
  "candidate_nonformal: $candidate_nonformal" \
  "surface_carrier_source: $([[ -n "$surface_carrier_root" ]] && printf true || printf false)" \
  "rejected_surface_carrier_ablation: $([[ "$allow_rejected_surface_carrier" == 1 ]] && printf true || printf false)" \
  "alpha_threshold: $alpha_threshold" \
  "ply_sha256: $(sha256sum "$observed_core/appearance_observed_canonical.ply" | awk '{print $1}')" \
  >"$run_dir/manifest.yaml"

input_args=(
  -m gaussian.prepare_vista4d_object_input
  --asset-root "$observed_core"
  --vksplat-root "$vksplat_root"
  --output "$artifact_dir"
  --width 672
  --height 384
  --fps 24
  --background 1.0
  --alpha-threshold "$alpha_threshold"
)
if [[ "$candidate_nonformal" == 1 ]]; then
  input_args+=(--candidate-nonformal)
fi
if [[ -n "$real_view_root" ]]; then
  [[ -f "$real_view_root/neutral_rgb/anchor_front.png" ]]
  [[ -f "$real_view_root/alpha/anchor_front.png" ]]
  input_args+=(--real-view-root "$real_view_root")
fi
if [[ -n "$surface_carrier_root" ]]; then
  [[ -f "$surface_carrier_root/manifest.json" ]]
  [[ -f "$surface_carrier_root/DONE" ]]
  input_args+=(
    --surface-carrier-root "$surface_carrier_root"
    --allow-rejected-surface-carrier-ablation
  )
fi
timeout --signal=TERM --kill-after=15 600 \
  "$python_bin" "${input_args[@]}" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

"$python_bin" - "$artifact_dir/input_manifest.json" "$artifact_dir/DONE" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
done = json.loads(Path(sys.argv[2]).read_text())
if manifest.get("schema_version") != "radeon_oneloop.vista4d_object_conditioning.v1":
    raise RuntimeError("unexpected conditioning schema")
if manifest.get("frames") != 49:
    raise RuntimeError("Vista4D requires 49 conditioning frames")
if manifest.get("formal") is not False or manifest.get("physical_output") is not False:
    raise RuntimeError("conditioning provenance boundary was weakened")
if done.get("status") != "complete":
    raise RuntimeError("conditioning bundle is incomplete")
PY

(cd "$run_dir" && find . -type f ! -name run_hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >run_hashes.sha256)
printf '{"status":"done_nonformal_vista4d_conditioning"}\n' >"$run_dir/DONE"
trap - EXIT
printf 'AMD Vista4D object conditioning passed: %s\n' "$run_dir"
