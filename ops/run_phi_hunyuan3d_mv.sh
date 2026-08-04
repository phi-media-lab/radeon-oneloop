#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <four-view-input-root> <output-root> <hunyuan-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
input_root=$1
output_root=$2
hunyuan_root=$3
python_bin=$4
seed=${ONELOOP_HUNYUAN_SEED:-10027}
steps=${ONELOOP_HUNYUAN_STEPS:-50}
guidance=${ONELOOP_HUNYUAN_GUIDANCE:-5.0}
octree=${ONELOOP_HUNYUAN_OCTREE:-380}
chunks=${ONELOOP_HUNYUAN_CHUNKS:-20000}
local_snapshot=${ONELOOP_HUNYUAN_LOCAL_SNAPSHOT:-}
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
run_id="hunyuan3d_2mv_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_seed${seed}"
run_dir="$output_root/$run_id"
artifact_dir="$run_dir/artifact"

[[ -d "$input_root" ]]
[[ -f "$input_root/manifest.json" ]]
[[ -f "$input_root/hashes.sha256" ]]
[[ -f "$input_root/DONE" ]]
[[ -d "$hunyuan_root/.git" ]]
[[ -x "$python_bin" ]]
[[ ! -e "$run_dir" ]]
mkdir -p "$run_dir"
snapshot_args=()
if [[ -n "$local_snapshot" ]]; then
  [[ -d "$local_snapshot" ]]
  snapshot_args=(--local-snapshot "$local_snapshot")
fi

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
    "schema_version": "radeon_oneloop.hunyuan3d_2mv_run_failure.v1",
    "stage": "MI300X_Hunyuan3D_2mv_complete_mesh_proposal",
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

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

git -C "$hunyuan_root" rev-parse HEAD >"$run_dir/hunyuan_commit.txt"
git -C "$hunyuan_root" status --porcelain=v1 >"$run_dir/hunyuan_git_status.txt"
{
  printf 'python=%s\n' "$python_bin"
  "$python_bin" -c 'import torch; print(f"torch={torch.__version__}"); print(f"hip={torch.version.hip}"); print(f"device={torch.cuda.get_device_name(0)}")'
  rocm-smi --showproductname --showmeminfo vram
} >"$run_dir/environment.txt"
printf '%q ' "$python_bin" -m gaussian.hunyuan3d_mv_generate \
  --input-root "$input_root" --output "$artifact_dir" \
  --seed "$seed" --num-inference-steps "$steps" --guidance-scale "$guidance" \
  --octree-resolution "$octree" --num-chunks "$chunks" --require-mi300x \
  "${snapshot_args[@]}" \
  >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

PYTHONPATH="$repo_root:$hunyuan_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=60 3600 \
  "$python_bin" -m gaussian.hunyuan3d_mv_generate \
  --input-root "$input_root" \
  --output "$artifact_dir" \
  --seed "$seed" \
  --num-inference-steps "$steps" \
  --guidance-scale "$guidance" \
  --octree-resolution "$octree" \
  --num-chunks "$chunks" \
  "${snapshot_args[@]}" \
  --require-mi300x \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

[[ -f "$artifact_dir/DONE" ]]
[[ -s "$artifact_dir/mesh/raw_hunyuan3d_2mv.glb" ]]
[[ -s "$artifact_dir/mesh/raw_hunyuan3d_2mv.ply" ]]

"$python_bin" - "$run_dir/manifest.json" "$run_id" "$artifact_dir" "$input_root" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, run_id, artifact_dir, input_root = sys.argv[1:]
artifact = Path(artifact_dir)
source = json.loads((artifact / "manifest.json").read_text())

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

value = {
    "schema_version": "radeon_oneloop.hunyuan3d_2mv_run.v1",
    "run_id": run_id,
    "formal": False,
    "stage": "MI300X_Hunyuan3D_2mv_complete_mesh_proposal",
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "input_manifest_sha256": sha256(Path(input_root) / "manifest.json"),
    "artifact_manifest_sha256": sha256(artifact / "manifest.json"),
    "artifact_hashes_sha256": sha256(artifact / "hashes.sha256"),
    "mesh_glb_sha256": source["mesh"]["glb_sha256"],
    "mesh_ply_sha256": source["mesh"]["ply_sha256"],
    "review_status": source["review_status"],
}
Path(manifest_path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
"$python_bin" - "$run_dir/DONE" "$run_dir/manifest.json" "$run_dir/hashes.sha256" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

done_path, manifest_path, hashes_path = map(Path, sys.argv[1:])
value = {
    "schema_version": "radeon_oneloop.hunyuan3d_2mv_run_done.v1",
    "stage": "MI300X_Hunyuan3D_2mv_complete_mesh_proposal",
    "status": "done_candidate_pending_alignment",
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "hashes_sha256": hashlib.sha256(hashes_path.read_bytes()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
done_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

trap - EXIT INT TERM
printf 'MI300X Hunyuan3D-2mv proposal complete: %s\n' "$run_dir"
