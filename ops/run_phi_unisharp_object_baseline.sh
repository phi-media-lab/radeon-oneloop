#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <m1-manifest> <pose-cameras> <output-root> <python-bin> <checkpoint> <unisharp-root>\n' "$0" >&2
  exit 64
}

[[ $# -eq 6 ]] || usage
m1_manifest=$1
pose_cameras=$2
output_root=$3
python_bin=$4
checkpoint=$5
unisharp_root=$6
infer_script="$unisharp_root/scripts/infer_unisharp.py"
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
capture_script="$repo_root/gaussian/unisharp_infer_with_frames.py"

[[ -f "$m1_manifest" ]]
[[ -f "$pose_cameras" ]]
[[ -x "$python_bin" ]]
[[ -f "$checkpoint" ]]
[[ -d "$unisharp_root/.git" ]]
[[ -f "$infer_script" ]]
[[ -f "$capture_script" ]]

script_path="$runner_dir/$(basename "${BASH_SOURCE[0]}")"
run_id="unisharp_m1_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
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
    "stage": "MI300X_UniSHARP_object_baseline",
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
    raise SystemExit("UniSHARP job requires exactly one visible ROCm device")
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

"$python_bin" - "$m1_manifest" "$pose_cameras" "$run_dir/cameras_unisharp_768.json" <<'PY'
import json, sys
from pathlib import Path
m1_path, cameras_path, output_path = map(Path, sys.argv[1:])
m1 = json.loads(m1_path.read_text())
cameras = json.loads(cameras_path.read_text())
views = {view["id"]: view for view in m1["views"] if view.get("prepared")}
entries = {}
for camera in cameras["cameras"]:
    view_id = camera["view_id"]
    if view_id not in views:
        raise SystemExit(f"pose camera is absent from M1: {view_id}")
    width, height = camera["image_size_wh"]
    if (width, height) != (512, 512):
        raise SystemExit(f"expected 512x512 VGGT camera, got {width}x{height}")
    scale = 768.0 / 512.0
    intrinsic = camera["intrinsic_3x3"]
    entries[f"{view_id}.png"] = {
        "camera": "perspective",
        "intrinsics": [
            intrinsic[0][0] * scale,
            intrinsic[1][1] * scale,
            intrinsic[0][2] * scale,
            intrinsic[1][2] * scale,
        ],
        "source_camera_view_id": view_id,
        "source_image_sha256": views[view_id]["neutral_image"]["sha256"],
    }
value = {
    "schema_version": "radeon_oneloop.unisharp_camera_inputs.v1",
    "formal": False,
    "input_resolution": [768, 768],
    "source_camera_resolution": [512, 512],
    "images": entries,
}
output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

unisharp_commit=$(git -C "$unisharp_root" rev-parse HEAD)
input_dir=$(dirname "$m1_manifest")/01_normalized/neutral_rgb
mapfile -t images < <(find "$input_dir" -maxdepth 1 -type f -name 'anchor_*.png' -printf '%f\n' | sort)
[[ ${#images[@]} -eq 4 ]]
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)
torch_lib="$unisharp_root/.venv/lib/python3.12/site-packages/torch/lib"
LD_LIBRARY_PATH="$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="$unisharp_root${PYTHONPATH:+:$PYTHONPATH}" \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
timeout --signal=TERM --kill-after=30 1800 \
  "$python_bin" "$capture_script" --infer-script "$infer_script" -- \
  --checkpoint "$checkpoint" \
  --image-dir "$input_dir" \
  --out-dir "$run_dir/inference" \
  --camera-json "$run_dir/cameras_unisharp_768.json" \
  --camera perspective \
  --save-ply \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
finished_epoch=$(date +%s)
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mapfile -t plys < <(find "$run_dir/inference" -type f -name gaussians.ply -printf '%p\n' | sort)
[[ ${#plys[@]} -eq 4 ]]

"$python_bin" - "$run_dir/manifest.json" "$run_id" "$m1_manifest" "$pose_cameras" \
  "$run_dir/cameras_unisharp_768.json" "$checkpoint" "$script_path" "$infer_script" "$capture_script" \
  "$unisharp_commit" "$hardware_json" "$started_utc" "$finished_utc" \
  "$((finished_epoch - started_epoch))" "$input_dir" "$run_dir/inference" <<'PY'
import hashlib, json, sys
from pathlib import Path
(
    manifest_path, run_id, m1_manifest, pose_cameras, generated_cameras,
    checkpoint, runner, infer_script, capture_script, commit, hardware_json, started_utc,
    finished_utc, runtime_s, input_dir, inference_dir,
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
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError(f"vertex count absent: {path}")

inputs = [
    {"basename": path.name, "sha256": sha256(path)}
    for path in sorted(Path(input_dir).glob("anchor_*.png"))
]
outputs = []
for path in sorted(Path(inference_dir).rglob("gaussians.ply")):
    metadata = json.loads(path.with_name("metadata.json").read_text())
    outputs.append({
        "view_id": Path(metadata["image"]).stem,
        "relpath": str(path.relative_to(Path(manifest_path).parent)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "gaussian_count": vertex_count(path),
        "metadata_sha256": sha256(path.with_name("metadata.json")),
    })
pseudoview_sets = []
for path in sorted(Path(inference_dir).rglob("pseudo_view_cameras.json")):
    document = json.loads(path.read_text())
    pseudoview_sets.append({
        "source_camera_view_id": document["source_camera_view_id"],
        "relpath": str(path.relative_to(Path(manifest_path).parent)),
        "sha256": sha256(path),
        "view_count": len(document["views"]),
    })
value = {
    "schema_version": "radeon_oneloop.unisharp_object_baseline.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "phi_amd_work_mi300x_nonformal_generator_to_gs",
    "model": "UniSHARP",
    "mode": "per_anchor_gaussian_and_lossless_pseudoview_hypotheses",
    "hardware": json.loads(hardware_json),
    "started_utc": started_utc,
    "finished_utc": finished_utc,
    "runtime_s": int(runtime_s),
    "m1_manifest_sha256": sha256(m1_manifest),
    "pose_cameras_sha256": sha256(pose_cameras),
    "generated_camera_json_sha256": sha256(generated_cameras),
    "checkpoint_sha256": sha256(checkpoint),
    "unisharp_commit": commit,
    "runner_sha256": sha256(runner),
    "inference_script_sha256": sha256(infer_script),
    "capture_script_sha256": sha256(capture_script),
    "inputs": inputs,
    "outputs": outputs,
    "pseudoview_sets": pseudoview_sets,
    "provenance": "generated_fill_candidate",
    "eligible_for_formal_metrics": False,
    "acceptance_status": "pending_cross_view_metric_fusion",
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
    "stage": "MI300X_UniSHARP_object_baseline",
    "status": "done_candidate",
    "manifest_sha256": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(done_path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'MI300X UniSHARP object baseline complete: %s\n' "$run_dir"
