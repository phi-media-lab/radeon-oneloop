#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <seva-run-root> <four-view-input-root> <output-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
seva_run_root=$1
four_view_input_root=$2
output_root=$3
python_bin=$4
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="seva_orbit_audit_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"

[[ -d "$seva_run_root" ]]
[[ -f "$seva_run_root/DONE" ]]
[[ -d "$four_view_input_root" ]]
[[ -x "$python_bin" ]]
mkdir -p "$output_root"
[[ ! -e "$run_dir" ]]

cd "$repo_root"
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m gaussian.audit_seva_orbit \
  --run-root "$seva_run_root" \
  --input-root "$four_view_input_root" \
  --output "$run_dir"

[[ -f "$run_dir/DONE" ]]
printf 'SEVA orbit audit complete: %s\n' "$run_dir"
