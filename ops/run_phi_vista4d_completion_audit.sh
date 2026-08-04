#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <conditioning-root> <proposal-run> <output-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
conditioning=$1
proposal=$2
output_root=$3
python_bin=$4
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="vista4d_audit_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"

mkdir -p "$output_root"
[[ ! -e "$run_dir" ]]
mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    mkdir -p "$run_dir"
    "$python_bin" - "$run_dir/FAILED" "$status" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, status = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema_version": "radeon_oneloop.vista4d_completion_visual_audit_failure.v1",
    "status": "failed", "exit_code": int(status),
    "failed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -d "$conditioning" ]]
[[ -d "$proposal" ]]
[[ -x "$python_bin" ]]
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=30 600 \
  "$python_bin" -m gaussian.audit_vista4d_completion \
  --conditioning "$conditioning" \
  --proposal-run "$proposal" \
  --output "$run_dir"

[[ -s "$run_dir/metrics.json" ]]
[[ -s "$run_dir/hashes.sha256" ]]
[[ -s "$run_dir/DONE" ]]
[[ $(find "$run_dir/generated_masks" -maxdepth 1 -type f -name '*.png' | wc -l) -eq 49 ]]
trap - EXIT INT TERM
printf 'Vista4D completion audit complete: %s\n' "$run_dir"
