#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <four-view-input-root> <output-root> <seva-root> <local-model-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 5 ]] || usage
input_root=$1
output_root=$2
seva_root=$3
local_model_root=$4
python_bin=$5
seed=${ONELOOP_SEVA_SEED:-10027}
model_revision=${ONELOOP_SEVA_MODEL_REVISION:-e538e251c1009e9a41cf8b7fee5f21332a1960de}
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id=${ONELOOP_SEVA_RUN_ID:-"seva_four_view_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_seed${seed}"}
run_dir="$output_root/$run_id"
source_output="$seva_root/work_dirs/demo/img2trajvid/$run_id/graffiti_mickey_four_view"

mkdir -p "$output_root"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]
[[ ! -e "$run_dir" ]]
[[ ! -e "$source_output" ]]
mkdir -p "$run_dir/inference"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    "$python_bin" - "$run_dir/FAILED" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status = sys.argv[1:]
value = {
    "schema_version": "radeon_oneloop.seva_four_view_run_failure.v1",
    "stage": "MI300X_SEVA_four_view_orbit_generation",
    "status": "failed",
    "exit_code": int(status),
    "failed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    (
      cd "$run_dir"
      find . -type f ! -name hashes.sha256 -print0 | sort -z | xargs -0 sha256sum >hashes.sha256
    )
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -d "$input_root" ]]
[[ -f "$input_root/manifest.json" ]]
[[ -d "$seva_root/.git" ]]
[[ -d "$local_model_root" ]]
[[ -s "$local_model_root/modelv1.1.safetensors" ]]
[[ -f "$local_model_root/config.yaml" ]]
[[ "$model_revision" =~ ^[0-9a-f]{40}$ ]]
[[ -x "$python_bin" ]]

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m gaussian.prepare_four_view_generation \
  --reviewed-root "$input_root" --output "$input_root" --validate-only >/dev/null

git -C "$seva_root" rev-parse HEAD >"$run_dir/seva_commit.txt"
git -C "$seva_root" status --porcelain=v1 >"$run_dir/seva_git_status.txt"
{
  printf 'python=%s\n' "$python_bin"
  "$python_bin" -c 'import torch,numpy; print(f"torch={torch.__version__}"); print(f"hip={torch.version.hip}"); print(f"numpy={numpy.__version__}"); print(f"device={torch.cuda.get_device_name(0)}")'
  rocm-smi --showproductname --showmeminfo vram
} >"$run_dir/environment.txt"
printf '%q ' "$python_bin" demo.py \
  --data_path "$input_root/seva" --data_items graffiti_mickey_four_view \
  --version 1.1 --task img2trajvid --save_subdir "$run_id" \
  --pretrained_model_name_or_path "$local_model_root" \
  --weight_name modelv1.1.safetensors \
  --num_inputs 4 --cfg 3.0,2.0 --use_traj_prior True \
  --chunk_strategy interp-gt --seed "$seed" --video_save_fps 12 \
  >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

started_epoch=$(date +%s)
cd "$seva_root"
PYTHONPATH="$seva_root:$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=60 3600 \
  "$python_bin" demo.py \
  --data_path "$input_root/seva" \
  --data_items graffiti_mickey_four_view \
  --version 1.1 \
  --task img2trajvid \
  --save_subdir "$run_id" \
  --pretrained_model_name_or_path "$local_model_root" \
  --weight_name modelv1.1.safetensors \
  --num_inputs 4 \
  --cfg 3.0,2.0 \
  --use_traj_prior True \
  --chunk_strategy interp-gt \
  --seed "$seed" \
  --video_save_fps 12 \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
runtime_s=$(($(date +%s) - started_epoch))

[[ -d "$source_output" ]]
cp -a "$source_output/." "$run_dir/inference/"

cd "$repo_root"
PYTHONPATH="$repo_root:$seva_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m gaussian.record_seva_four_view_run \
  --run-dir "$run_dir" \
  --input-root "$input_root" \
  --seva-root "$seva_root" \
  --local-model-root "$local_model_root" \
  --revision "$model_revision" \
  --seed "$seed" \
  --runtime-s "$runtime_s" \
  >>"$run_dir/stdout.log" 2>>"$run_dir/stderr.log"

[[ -f "$run_dir/DONE" ]]
[[ -s "$run_dir/inference/samples-rgb.mp4" ]]
trap - EXIT INT TERM
printf 'MI300X SEVA four-view orbit complete: %s\n' "$run_dir"
