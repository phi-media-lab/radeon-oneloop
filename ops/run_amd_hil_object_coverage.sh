#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_AMD_PYTHON:-/home/amd/.venvs/drtc-rocm/bin/python}
dataset=${ONELOOP_HIL_DATASET:?set ONELOOP_HIL_DATASET to the private LeRobot v3 HIL dataset}
run_root=${ONELOOP_AMD_RUN_ROOT:-/home/amd/radeon-oneloop-runs/real2sim/hil_object_coverage}
phase_samples=${ONELOOP_HIL_PHASE_SAMPLES:-12}

created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
run_id=$(date -u +%Y%m%dT%H%M%SZ)_$$_amd_hil_object_coverage
run_dir="$run_root/$run_id"
mkdir -p "$run_dir"

command=(
  "$python_bin" -m gaussian.hil_object_coverage
  --dataset "$dataset"
  --output "$run_dir/artifact"
  --phase-samples "$phase_samples"
  --created-utc "$created_utc"
)
printf '%q ' "${command[@]}" >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

(
  cd "$repo_root"
  PYTHONPATH="$repo_root" "${command[@]}"
) > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2) || {
  touch "$run_dir/FAILED"
  exit 1
}

(
  cd "$run_dir"
  sha256sum command.sh stdout.log stderr.log artifact/manifest.json \
    artifact/hashes.sha256 artifact/DONE >hashes.sha256
)
mv "$run_dir/artifact/DONE" "$run_dir/artifact/.DONE.inner"
touch "$run_dir/DONE"
mv "$run_dir/artifact/.DONE.inner" "$run_dir/artifact/DONE"
printf 'HIL object coverage audit complete: %s\n' "$run_dir"
