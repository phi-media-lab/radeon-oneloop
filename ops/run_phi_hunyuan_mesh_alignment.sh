#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <four-view-input-root> <hunyuan-run-root> <output-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
input_root=$1
hunyuan_run=$2
output_root=$3
python_bin=$4
target_faces=${ONELOOP_ALIGNMENT_TARGET_FACES:-60000}
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="hunyuan_alignment_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"

[[ -d "$input_root" ]]
[[ -d "$hunyuan_run" ]]
[[ -x "$python_bin" ]]
[[ ! -e "$run_dir" ]]

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=30 1200 \
  "$python_bin" -m gaussian.align_generated_mesh_four_views \
  --input-root "$input_root" \
  --hunyuan-run "$hunyuan_run" \
  --output "$run_dir" \
  --target-faces "$target_faces"

[[ -f "$run_dir/DONE" ]]
[[ -s "$run_dir/mesh/aligned_metric_hunyuan3d_2mv.glb" ]]
[[ -s "$run_dir/audit/four_real_view_alignment_contact_sheet.png" ]]
printf 'Four-view learned-mesh coarse alignment complete: %s\n' "$run_dir"
