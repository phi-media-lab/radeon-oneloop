#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <input-dir> <m1-manifest> <output-root> <python-bin> <sharp-bin> <checkpoint>\n' "$0" >&2
  exit 64
}

[[ $# -eq 6 ]] || usage
input_dir=$1
m1_manifest=$2
output_root=$3
python_bin=$4
sharp_bin=$5
checkpoint=$6

[[ -d "$input_dir" ]]
[[ -f "$m1_manifest" ]]
[[ -x "$python_bin" ]]
[[ -x "$sharp_bin" ]]
[[ -f "$checkpoint" ]]

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
run_id="sharp_m1_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"
ply_dir="$run_dir/ply"
mkdir -p "$ply_dir"

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    "$python_bin" - "$run_dir/FAILED" "$status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status = sys.argv[1:]
value = {
    "stage": "MI300X_SHARP_object_baseline",
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

mapfile -t images < <(find "$input_dir" -maxdepth 1 -type f -name 'anchor_*.png' -printf '%f\n' | sort)
if [[ ${#images[@]} -ne 4 ]]; then
  printf 'expected four anchor PNGs, found %d\n' "${#images[@]}" >&2
  exit 65
fi

hardware_json=$(
  "$python_bin" - <<'PY'
import json, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("SHARP job requires exactly one visible ROCm device")
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

started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)
timeout --signal=TERM --kill-after=20 900 \
  "$sharp_bin" predict \
  --input-path "$input_dir" \
  --output-path "$ply_dir" \
  --checkpoint-path "$checkpoint" \
  --device cuda \
  --no-render \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
finished_epoch=$(date +%s)
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mapfile -t outputs < <(find "$ply_dir" -maxdepth 1 -type f -name '*.ply' -printf '%f\n' | sort)
if [[ ${#outputs[@]} -ne 4 ]]; then
  printf 'SHARP produced %d PLY files, expected four\n' "${#outputs[@]}" >&2
  exit 65
fi

"$python_bin" - \
  "$run_dir/manifest.json" "$run_id" "$input_dir" "$m1_manifest" \
  "$checkpoint" "$script_path" "$started_utc" "$finished_utc" \
  "$((finished_epoch - started_epoch))" "$hardware_json" "$ply_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    manifest_path,
    run_id,
    input_dir,
    m1_manifest,
    checkpoint,
    script_path,
    started_utc,
    finished_utc,
    runtime_s,
    hardware_json,
    ply_dir,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def vertex_count(path):
    with open(path, "rb") as stream:
        for _ in range(256):
            line = stream.readline()
            if not line:
                break
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("element vertex "):
                return int(text.rsplit(" ", 1)[1])
            if text == "end_header":
                break
    raise ValueError(f"PLY vertex count absent: {path}")

input_root = Path(input_dir)
output_root = Path(ply_dir)
inputs = [
    {"basename": path.name, "sha256": sha256(path)}
    for path in sorted(input_root.glob("anchor_*.png"))
]
outputs = [
    {
        "basename": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "gaussian_count": vertex_count(path),
    }
    for path in sorted(output_root.glob("*.ply"))
]
value = {
    "schema_version": "radeon_oneloop.sharp_object_baseline.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "phi_amd_work_mi300x_nonformal_generator_to_gs",
    "model": "apple_ml_sharp",
    "mode": "per_anchor_gaussian_hypotheses",
    "started_utc": started_utc,
    "finished_utc": finished_utc,
    "runtime_s": int(runtime_s),
    "hardware": json.loads(hardware_json),
    "m1_manifest_sha256": sha256(m1_manifest),
    "checkpoint_sha256": sha256(checkpoint),
    "runner_sha256": sha256(script_path),
    "inputs": inputs,
    "outputs": outputs,
    "provenance": "generated_fill_candidate",
    "acceptance_status": "pending_cross_view_pose_and_consistency_gate",
}
Path(manifest_path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
"$python_bin" - "$run_dir/DONE" "$run_dir/manifest.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
done_path, manifest_path = sys.argv[1:]
digest = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
value = {
    "stage": "MI300X_SHARP_object_baseline",
    "status": "done",
    "manifest_sha256": digest,
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(done_path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'MI300X SHARP object baseline passed: %s\n' "$run_dir"
