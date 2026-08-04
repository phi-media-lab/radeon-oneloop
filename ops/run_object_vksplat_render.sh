#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <ply> <cameras> <source-provenance> <output-root> <python-bin> <vksplat-root> <repo-root>\n' "$0" >&2
  exit 64
}

[[ $# -eq 7 ]] || usage
ply=$1
cameras=$2
source_provenance=$3
output_root=$4
python_bin=$5
vksplat_root=$6
repo_root=$7
expected_camera_count=${ONELOOP_VKSPLAT_EXPECTED_CAMERAS:-4}
background=${ONELOOP_VKSPLAT_BACKGROUND:-1.0}
render_script="$repo_root/gaussian/vksplat_render_ply.py"

[[ -f "$ply" ]]
[[ -f "$cameras" ]]
[[ -f "$source_provenance" ]]
[[ -x "$python_bin" ]]
[[ -f "$vksplat_root/vksplat/simple_trainer.py" ]]
[[ -f "$render_script" ]]
[[ "$expected_camera_count" =~ ^[1-9][0-9]*$ ]]
"$python_bin" - "$background" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit("render background must be finite and in [0, 1]")
PY

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
run_id="vksplat_generated_fill_render_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"
mkdir -p "$run_dir"

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    "$python_bin" - "$run_dir/FAILED" "$status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status = sys.argv[1:]
value = {
    "stage": "radeon_f_VkSplat_generated_fill_render",
    "status": "failed",
    "exit_code": int(status),
    "failed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

vulkan_summary=$(vulkaninfo --summary 2>/dev/null)
if ! grep -q 'vendorID.*0x1002' <<<"$vulkan_summary" || ! grep -q 'RADV NAVI31' <<<"$vulkan_summary"; then
  printf '%s\n' 'expected a RADV NAVI31 Radeon device' >&2
  exit 69
fi
printf '%s\n' "$vulkan_summary" >"$run_dir/vulkaninfo_summary.txt"
vksplat_commit=${vksplat_root##*/vksplat-}
[[ "$vksplat_commit" =~ ^[0-9a-f]{40}$ ]]
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"

XDG_RUNTIME_DIR="$xdg_runtime" timeout --signal=TERM --kill-after=15 300 \
  "$python_bin" "$render_script" \
  --ply "$ply" \
  --cameras "$cameras" \
  --source-provenance "$source_provenance" \
  --output "$run_dir/render" \
  --vksplat-root "$vksplat_root" \
  --vksplat-commit "$vksplat_commit" \
  --expected-camera-count "$expected_camera_count" \
  --background "$background" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

"$python_bin" - "$run_dir/manifest.json" "$run_id" "$ply" "$cameras" \
  "$source_provenance" "$script_path" "$render_script" "$vksplat_commit" \
  "$run_dir/render/render_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
(
    manifest_path, run_id, ply, cameras, source_provenance, runner,
    render_script, vksplat_commit, render_manifest,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

render = json.loads(Path(render_manifest).read_text())
value = {
    "schema_version": "radeon_oneloop.vksplat_generated_fill_render_run.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "radeon_f_nonformal_RADV_renderer",
    "renderer": "VkSplat",
    "vksplat_commit": vksplat_commit,
    "ply_sha256": sha256(ply),
    "cameras_sha256": sha256(cameras),
    "source_provenance_sha256": sha256(source_provenance),
    "runner_sha256": sha256(runner),
    "render_script_sha256": sha256(render_script),
    "render_manifest_sha256": sha256(render_manifest),
    "gaussian_count": render["gaussian_count"],
    "acceptance_status": "pending_visual_generated_fill_review",
    "eligible_for_formal_metrics": False,
}
Path(manifest_path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
"$python_bin" - "$run_dir/DONE" "$run_dir/manifest.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
done_path, manifest_path = sys.argv[1:]
value = {
    "stage": "radeon_f_VkSplat_generated_fill_render",
    "status": "done_nonformal_visual_candidate",
    "manifest_sha256": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(done_path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'Radeon-f VkSplat generated-fill render complete: %s\n' "$run_dir"
