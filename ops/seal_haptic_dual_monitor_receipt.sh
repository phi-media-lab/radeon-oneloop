#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s SOURCE_DUAL_MONITOR_RUN_DIR MAPPING_VERDICT\n' "$0" >&2
  printf 'mapping: both_correct_parallel_same_direction|left_right_swapped_or_direction_wrong|unstable_or_cross_coupled\n' >&2
  exit 64
fi

source_run_dir=$1
mapping_verdict=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_LEROBOT_PYTHON:-${ONELOOP_VALIDATION_PYTHON:-python3}}
receipt_root=${ONELOOP_HAPTIC_RECEIPT_ROOT:-$repo_root/runs/haptic_stage_receipts}
leaders_free=${ONELOOP_BOTH_LEADERS_MOVE_FREELY_CONFIRMED:-0}
source_run_id=$(basename "$source_run_dir")
receipt_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_dual_arm_monitor_receipt"
receipt_dir="$receipt_root/$receipt_id"

if [[ "$python_bin" == */* ]]; then
  [[ -x "$python_bin" ]]
else
  command -v "$python_bin" >/dev/null
fi
[[ -f "$source_run_dir/gate.json" ]]
[[ -f "$source_run_dir/hashes.sha256" ]]
[[ -f "$source_run_dir/DONE" ]]
[[ "$leaders_free" == 1 ]]
case "$mapping_verdict" in
  both_correct_parallel_same_direction|left_right_swapped_or_direction_wrong|unstable_or_cross_coupled) ;;
  *) exit 64 ;;
esac
mkdir -p "$receipt_dir"

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$receipt_dir/DONE" ]]; then
    files=("$receipt_dir/manifest.yaml")
    if [[ -f "$receipt_dir/receipt.json" ]]; then
      files+=("$receipt_dir/receipt.json")
    fi
    sha256sum "${files[@]}" >"$receipt_dir/hashes.sha256"
    printf '{"status":"failed_or_operator_rejected","exit_code":%d}\n' \
      "$status" >"$receipt_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT

printf '%s\n' \
  'schema_version: radeon_oneloop.haptic_dual_monitor_receipt_run.v1' \
  'formal: false' \
  'stage: dual_arm_monitor_only' \
  "source_run_id: $source_run_id" \
  "operator_mapping_verdict: $mapping_verdict" \
  'operator_identity_recorded: false' \
  'physical_output_commands: false' \
  >"$receipt_dir/manifest.yaml"

export PYTHONPATH="$repo_root/src:$repo_root"
"$python_bin" -m sim.genesis_so101.haptic_dual_monitor_receipt \
  --source-run-id "$source_run_id" \
  --gate "$source_run_dir/gate.json" \
  --source-hash-index "$source_run_dir/hashes.sha256" \
  --source-done "$source_run_dir/DONE" \
  --mapping-verdict "$mapping_verdict" \
  --both-leaders-move-freely \
  --output "$receipt_dir/receipt.json"

sha256sum "$receipt_dir/manifest.yaml" "$receipt_dir/receipt.json" \
  >"$receipt_dir/hashes.sha256"
sha256sum -c "$receipt_dir/hashes.sha256" >/dev/null
printf '{"status":"done_dual_arm_monitor_stage_accepted"}\n' \
  >"$receipt_dir/DONE"
trap - EXIT
printf 'dual-arm monitor receipt sealed: %s\n' "$receipt_dir"
