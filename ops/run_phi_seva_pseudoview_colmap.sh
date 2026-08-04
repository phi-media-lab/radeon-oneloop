#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <four-view> <seva-run> <audit> <review> <observed-initialization> <output-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 7 ]] || usage
four_view=$1
seva_run=$2
audit=$3
review=$4
observed_initialization=$5
output_root=$6
python_bin=$7
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="seva_pseudoview_colmap_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"

mkdir -p "$output_root"
[[ ! -e "$run_dir" ]]
[[ -d "$four_view" && -d "$seva_run" && -d "$audit" && -d "$observed_initialization" ]]
[[ -f "$review" && -x "$python_bin" ]]

cd "$repo_root"
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=30 900 \
  "$python_bin" -m gaussian.seva_pseudoview_colmap \
  --four-view-input "$four_view" \
  --seva-run "$seva_run" \
  --audit "$audit" \
  --review "$review" \
  --observed-initialization "$observed_initialization" \
  --output "$run_dir" \
  --real-repeat 12 \
  --generated-count 24 \
  --max-points 30000 \
  --sample-seed 20260804

[[ -s "$run_dir/dataset_manifest.json" ]]
[[ -s "$run_dir/hashes.sha256" ]]
[[ -s "$run_dir/DONE" ]]
[[ $(find "$run_dir/images" -maxdepth 1 -type f -name '*.png' | wc -l) -eq 73 ]]
[[ $(find "$run_dir/masks" -maxdepth 1 -type f -name '*.png' | wc -l) -eq 73 ]]
printf 'SEVA pseudo-view COLMAP dataset complete: %s\n' "$run_dir"
