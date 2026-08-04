#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <four-view-input-root> <alignment-root> <output-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
input_root=$1
alignment_root=$2
output_root=$3
python_bin=$4
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="learned_mesh_texture_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"

mkdir -p "$output_root"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    "$python_bin" - "$run_dir.FAILED.json" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status = sys.argv[1:]
value = {
    "schema_version": "radeon_oneloop.four_view_learned_mesh_texture_orbit_failure.v1",
    "stage": "four_real_view_texture_projection_and_orbit_render",
    "status": "failed",
    "exit_code": int(status),
    "failed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -d "$input_root" ]]
[[ -d "$alignment_root" ]]
[[ -x "$python_bin" ]]
[[ ! -e "$run_dir" ]]

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=30 1200 \
  "$python_bin" -m gaussian.texture_learned_mesh_four_views \
  --input-root "$input_root" \
  --alignment-root "$alignment_root" \
  --output "$run_dir" \
  --visibility-tolerance-m 0.002 \
  --width 672 --height 384 --fps 12 \
  --distance-m 0.24 --horizontal-fov-deg 50

[[ -f "$run_dir/DONE" ]]
[[ -s "$run_dir/orbit/source.mp4" ]]
[[ -s "$run_dir/audit/orbit_contact_sheet.png" ]]
trap - EXIT INT TERM
printf 'Four-view learned-mesh texture orbit complete: %s\n' "$run_dir"
