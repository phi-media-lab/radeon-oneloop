#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <four-view> <texture> <conditioning> <observed-initialization> <output-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 6 ]] || usage
four_view=$1
texture=$2
conditioning=$3
observed_initialization=$4
output_root=$5
python_bin=$6
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="learned_mesh_hybrid_colmap_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"

mkdir -p "$output_root"
[[ ! -e "$run_dir" ]]
[[ -d "$four_view" && -d "$texture" && -d "$conditioning" && -d "$observed_initialization" ]]
[[ -x "$python_bin" ]]
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=30 900 \
  "$python_bin" -m gaussian.hybrid_pseudoview_colmap \
  --source-mode learned_mesh_orbit \
  --four-view-input "$four_view" \
  --texture-root "$texture" \
  --conditioning "$conditioning" \
  --observed-initialization "$observed_initialization" \
  --output "$run_dir" \
  --real-repeat 12 \
  --real-camera-distance-m 1.0 \
  --max-points 30000 \
  --sample-seed 20260804

[[ -s "$run_dir/dataset_manifest.json" ]]
[[ -s "$run_dir/hashes.sha256" ]]
[[ -s "$run_dir/DONE" ]]
[[ $(find "$run_dir/images" -maxdepth 1 -type f -name '*.png' | wc -l) -eq 98 ]]
[[ $(find "$run_dir/masks" -maxdepth 1 -type f -name '*.png' | wc -l) -eq 98 ]]
printf 'Learned-mesh hybrid COLMAP dataset complete: %s\n' "$run_dir"
