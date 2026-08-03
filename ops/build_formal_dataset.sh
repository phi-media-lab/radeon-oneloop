#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-/root/radeon-oneloop/current}
output=${2:-/root/radeon-oneloop-data/formal_handover_v1}
source_root=${3:-/root/radeon-oneloop-data/sources}
run_root=${4:-/root/radeon-oneloop-runs/dataset}
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}

[[ -x $python_bin ]]
[[ -d $project_root/.git ]]
commit=$(git -C "$project_root" rev-parse HEAD)
run_dir="$run_root/$(date -u +%Y%m%dT%H%M%SZ)_formal_dataset_${commit:0:7}"
mkdir -p "$run_dir"
exec > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)
trap 'code=$?; if [[ $code -ne 0 ]]; then touch "$run_dir/FAILED"; fi' EXIT

if [[ -e $output ]]; then
  printf 'refusing to overwrite existing dataset: %s\n' "$output" >&2
  exit 73
fi
export PYTHONPATH="$project_root/src:$project_root${PYTHONPATH:+:$PYTHONPATH}"
bc="$source_root/bc_seed"
hil="$source_root/hil_batch1_batch2"
manifest="$hil/makermods_hil/combined_hil_batch1_batch2_phase_aware_awr_v2_20260507/handover_rl_seed_manifest_v0.jsonl"

printf '%s\n' \
  "$python_bin -m radeon_oneloop.data_merge --bc-root $bc --hil-root $hil --hil-manifest $manifest --output $output" \
  > "$run_dir/command.sh"
chmod +x "$run_dir/command.sh"
"$python_bin" -m radeon_oneloop.data_merge \
  --bc-root "$bc" \
  --hil-root "$hil" \
  --hil-manifest "$manifest" \
  --output "$output" \
  | tee "$run_dir/dataset_report_stdout.json"

"$python_bin" -m radeon_oneloop.phase_targets \
  --dataset-parquet "$output/data/chunk-000/file-000.parquet" \
  --episode-manifest "$output/oneloop/episode_manifest.jsonl" \
  --output-parquet "$output/oneloop/phase_targets.parquet" \
  --report "$output/oneloop/phase_targets_report.json" \
  | tee "$run_dir/phase_targets_stdout.json"

"$python_bin" - "$run_dir/manifest.json" "$commit" "$output" <<'PY'
import json
import sys
from pathlib import Path

path, commit, output = sys.argv[1:]
root = Path(output)
dataset = json.loads((root / "oneloop/dataset_report.json").read_text())
targets = json.loads((root / "oneloop/phase_targets_report.json").read_text())
assert dataset["episodes"] == 124
assert dataset["frames"] == 178465
assert targets["summary"]["episodes"] == 124
assert targets["summary"]["frames"] == 178465
assert targets["summary"]["roles"]["correction"]["frames"] > 0
value = {
    "schema_version": "radeon_oneloop.dataset_build.v1",
    "git_commit": commit,
    "status": "done",
    "output": str(root),
    "dataset_hash": dataset["dataset_hash"],
    "phase_targets_sha256": targets["output"]["targets_sha256"],
    "episodes": dataset["episodes"],
    "frames": dataset["frames"],
}
Path(path).write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(value, indent=2))
PY

cp "$output/oneloop/dataset_report.json" "$run_dir/dataset_report.json"
cp "$output/oneloop/phase_targets_report.json" "$run_dir/phase_targets_report.json"
sha256sum "$run_dir"/*.json "$run_dir"/*.log "$run_dir/command.sh" \
  > "$run_dir/hashes.sha256"
touch "$run_dir/DONE"
trap - EXIT
printf 'formal dataset build passed: %s\n' "$run_dir"
