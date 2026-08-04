#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <texture-root> <four-view-input-root> <visual-review-json> <output-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 5 ]] || usage
texture_root=$1
four_view_root=$2
review=$3
output_root=$4
python_bin=$5
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="vista4d_learned_mesh_input_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"

[[ -d "$texture_root" ]]
[[ -d "$four_view_root" ]]
[[ -f "$review" ]]
[[ -x "$python_bin" ]]
[[ ! -e "$run_dir" ]]

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=30 600 \
  "$python_bin" -m gaussian.prepare_vista4d_learned_mesh_input \
  --texture-root "$texture_root" \
  --four-view-input "$four_view_root" \
  --review "$review" \
  --output "$run_dir" \
  --fps 24

[[ -s "$run_dir/input_manifest.json" ]]
[[ -s "$run_dir/hashes.sha256" ]]
[[ -s "$run_dir/DONE" ]]
printf 'Vista4D learned-mesh conditioning complete: %s\n' "$run_dir"
