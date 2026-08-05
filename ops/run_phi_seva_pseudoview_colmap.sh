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
failed_dir="$run_dir.FAILED"

mkdir -p "$output_root"
[[ ! -e "$run_dir" ]]
[[ ! -e "$failed_dir" ]]
[[ -d "$four_view" && -d "$seva_run" && -d "$audit" && -d "$observed_initialization" ]]
[[ -f "$review" && -x "$python_bin" ]]

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    if [[ ! -d "$failed_dir" ]]; then
      mkdir -p "$failed_dir"
    fi
    "$python_bin" - "$failed_dir/WRAPPER_FAILED.json" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "radeon_oneloop.seva_pseudoview_colmap_wrapper_failure.v1",
    "formal": False,
    "status": "failed",
    "exit_code": int(sys.argv[2]),
    "failed_utc": datetime.now(timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z"),
}, indent=2, sort_keys=True) + "\n")
PY
    (
      cd "$failed_dir"
      find . -type f ! -name hashes.sha256 -print0 | sort -z \
        | xargs -0 sha256sum >hashes.sha256
    )
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
trap - EXIT INT TERM
printf 'SEVA pseudo-view COLMAP dataset complete: %s\n' "$run_dir"
