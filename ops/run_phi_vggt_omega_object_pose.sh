#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <m1-manifest> <output-root> <python-bin> <checkpoint> <vggt-omega-root> <repo-root>\n' "$0" >&2
  exit 64
}

[[ $# -eq 6 ]] || usage
m1_manifest=$1
output_root=$2
python_bin=$3
checkpoint=$4
vggt_omega_root=$5
repo_root=$6
pose_script="$repo_root/gaussian/vggt_omega_object_pose.py"
geometry_script="$repo_root/gaussian/object_pose_init.py"

[[ -f "$m1_manifest" ]]
[[ -x "$python_bin" ]]
[[ -f "$checkpoint" ]]
[[ -d "$vggt_omega_root/.git" ]]
[[ -f "$pose_script" ]]
[[ -f "$geometry_script" ]]

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
run_id="vggt_omega_m1_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
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
    "stage": "MI300X_VGGT_Omega_object_pose",
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

hardware_json=$(
  "$python_bin" - <<'PY'
import json, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("VGGT-Omega job requires exactly one visible ROCm device")
name = torch.cuda.get_device_name(0)
if "MI300X" not in name:
    raise SystemExit(f"expected MI300X, got {name}")
print(json.dumps({
    "device": name,
    "device_count": torch.cuda.device_count(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
}))
PY
)
vggt_commit=$(git -C "$vggt_omega_root" rev-parse HEAD)
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" timeout --signal=TERM --kill-after=30 1800 \
  "$python_bin" "$pose_script" \
  --m1-manifest "$m1_manifest" \
  --output "$run_dir/pose_candidate" \
  --vggt-omega-root "$vggt_omega_root" \
  --checkpoint "$checkpoint" \
  --vggt-omega-commit "$vggt_commit" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
finished_epoch=$(date +%s)
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

"$python_bin" - \
  "$run_dir/manifest.json" "$run_id" "$m1_manifest" "$checkpoint" \
  "$script_path" "$pose_script" "$geometry_script" "$vggt_commit" \
  "$started_utc" "$finished_utc" "$((finished_epoch - started_epoch))" \
  "$hardware_json" "$run_dir/pose_candidate" <<'PY'
import hashlib, json, sys
from pathlib import Path

(
    manifest_path, run_id, m1_manifest, checkpoint, runner, pose_script,
    geometry_script, vggt_commit, started_utc, finished_utc, runtime_s,
    hardware_json, output_dir,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

output_root = Path(output_dir)
summary = json.loads((output_root / "inference_summary.json").read_text())
outputs = []
for path in sorted(item for item in output_root.iterdir() if item.is_file()):
    outputs.append({
        "relpath": f"pose_candidate/{path.name}",
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    })
value = {
    "schema_version": "radeon_oneloop.vggt_omega_object_pose_run.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "phi_amd_work_mi300x_nonformal_pose_and_depth_initializer",
    "model": "VGGT-Omega-1B-512",
    "vggt_omega_commit": vggt_commit,
    "started_utc": started_utc,
    "finished_utc": finished_utc,
    "runtime_s": int(runtime_s),
    "hardware": json.loads(hardware_json),
    "m1_manifest_sha256": sha256(m1_manifest),
    "checkpoint_sha256": sha256(checkpoint),
    "runner_sha256": sha256(runner),
    "pose_script_sha256": sha256(pose_script),
    "geometry_script_sha256": sha256(geometry_script),
    "outputs": outputs,
    "numeric_gate_passed": summary["quality"]["numeric_gate_passed"],
    "acceptance_status": summary["quality"]["acceptance_status"],
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
    "stage": "MI300X_VGGT_Omega_object_pose",
    "status": "done_candidate",
    "manifest_sha256": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(done_path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'MI300X VGGT-Omega object pose candidate complete: %s\n' "$run_dir"
