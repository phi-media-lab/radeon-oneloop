#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <m1-manifest> <pose-run> <pose-audit-manifest> <geometry-run> <output-root> <python-bin> <repo-root> [appearance-run]\n' "$0" >&2
  exit 64
}

[[ $# -eq 7 || $# -eq 8 ]] || usage
m1_manifest=$1
pose_run=$2
pose_audit_manifest=$3
sharp_run=$4
output_root=$5
python_bin=$6
repo_root=$7
appearance_run=${8:-}
fusion_script="$repo_root/gaussian/sharp_object_fusion.py"

[[ -f "$m1_manifest" ]]
[[ -d "$pose_run" ]]
[[ -f "$pose_audit_manifest" ]]
[[ -d "$sharp_run" ]]
[[ -x "$python_bin" ]]
[[ -f "$fusion_script" ]]
if [[ -n "$appearance_run" ]]; then
  [[ -d "$appearance_run" ]]
  [[ -f "$appearance_run/manifest.json" ]]
fi
appearance_manifest=${appearance_run:+$appearance_run/manifest.json}
appearance_manifest=${appearance_manifest:-none}
min_source_views=${MIN_SOURCE_VIEWS:-2}
min_silhouette_views=${MIN_SILHOUETTE_VIEWS:-2}
max_fused_gaussians=${MAX_FUSED_GAUSSIANS:-300000}
reduction_policy=${REDUCTION_POLICY:-keep_cross_source_supported}
[[ "$min_source_views" =~ ^[1-4]$ ]]
[[ "$min_silhouette_views" =~ ^[1-4]$ ]]
[[ "$max_fused_gaussians" =~ ^[1-9][0-9]*$ ]]
[[ "$reduction_policy" == keep_cross_source_supported || "$reduction_policy" == best_per_voxel ]]

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
run_id="sharp_fusion_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"
mkdir -p "$run_dir"
hardware_json='{}'
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    "$python_bin" - "$run_dir/manifest.json" "$run_id" "$status" "$m1_manifest" \
      "$pose_run/manifest.json" "$pose_audit_manifest" "$sharp_run/manifest.json" \
      "$script_path" "$fusion_script" "$hardware_json" "$started_utc" "$failed_utc" \
      "$appearance_manifest" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
(
    path, run_id, status, m1_manifest, pose_manifest, pose_audit, generator_manifest,
    runner, fusion_script, hardware_json, started_utc, failed_utc, appearance_manifest,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

generator = json.loads(Path(generator_manifest).read_text())
appearance = json.loads(Path(appearance_manifest).read_text()) if appearance_manifest != "none" else None
value = {
    "schema_version": "radeon_oneloop.sharp_metric_fusion_run.v1",
    "run_id": run_id,
    "formal": False,
    "stage": "MI300X_SHARP_metric_cross_view_fusion",
    "status": "failed",
    "exit_code": int(status),
    "started_utc": started_utc,
    "failed_utc": failed_utc,
    "hardware": json.loads(hardware_json),
    "model_chain": [
        generator.get("model"),
        *([appearance.get("model")] if appearance is not None else []),
        "VGGT-Omega-1B-512",
    ],
    "m1_manifest_sha256": sha256(m1_manifest),
    "pose_manifest_sha256": sha256(pose_manifest),
    "pose_visual_audit_manifest_sha256": sha256(pose_audit),
    "generator_manifest_sha256": sha256(generator_manifest),
    "appearance_manifest_sha256": (
        sha256(appearance_manifest) if appearance_manifest != "none" else None
    ),
    "runner_sha256": sha256(runner),
    "fusion_script_sha256": sha256(fusion_script),
    "eligible_for_formal_metrics": False,
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    (cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
      -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
    "$python_bin" - "$run_dir/FAILED" "$run_dir/manifest.json" "$status" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
path, manifest_path, status = sys.argv[1:]
value = {
    "stage": "MI300X_SHARP_metric_cross_view_fusion",
    "status": "failed",
    "exit_code": int(status),
    "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    "failed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    raise SystemExit("fusion job requires exactly one visible ROCm device")
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

fusion_args=(
  "$python_bin" "$fusion_script"
  --m1-manifest "$m1_manifest"
  --pose-run "$pose_run"
  --pose-audit-manifest "$pose_audit_manifest"
  --sharp-run "$sharp_run"
  --output "$run_dir/fusion"
  --min-source-views "$min_source_views"
  --min-silhouette-views "$min_silhouette_views"
  --max-fused-gaussians "$max_fused_gaussians"
  --reduction-policy "$reduction_policy"
)
if [[ -n "$appearance_run" ]]; then
  fusion_args+=(--appearance-run "$appearance_run")
fi
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" timeout --signal=TERM --kill-after=30 1800 \
  "${fusion_args[@]}" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
finished_epoch=$(date +%s)
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

"$python_bin" - \
  "$run_dir/manifest.json" "$run_id" "$m1_manifest" "$pose_run/manifest.json" \
  "$pose_audit_manifest" "$sharp_run/manifest.json" "$script_path" "$fusion_script" \
  "$started_utc" "$finished_utc" "$((finished_epoch - started_epoch))" \
  "$hardware_json" "$run_dir/fusion" "$appearance_manifest" <<'PY'
import hashlib, json, sys
from pathlib import Path
(
    manifest_path, run_id, m1_manifest, pose_manifest, pose_audit, sharp_manifest,
    runner, fusion_script, started_utc, finished_utc, runtime_s, hardware_json,
    output_dir, appearance_manifest,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

output = Path(output_dir)
quality = json.loads((output / "quality.json").read_text())
generator = json.loads(Path(sharp_manifest).read_text())
appearance = json.loads(Path(appearance_manifest).read_text()) if appearance_manifest != "none" else None
files = [
    {"relpath": f"fusion/{path.name}", "sha256": sha256(path), "bytes": path.stat().st_size}
    for path in sorted(output.iterdir()) if path.is_file()
]
value = {
    "schema_version": "radeon_oneloop.sharp_metric_fusion_run.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "phi_amd_work_mi300x_nonformal_generator_to_gs_fusion",
    "model_chain": [
        generator["model"],
        *([appearance["model"]] if appearance is not None else []),
        "VGGT-Omega-1B-512",
    ],
    "started_utc": started_utc,
    "finished_utc": finished_utc,
    "runtime_s": int(runtime_s),
    "hardware": json.loads(hardware_json),
    "m1_manifest_sha256": sha256(m1_manifest),
    "pose_manifest_sha256": sha256(pose_manifest),
    "pose_visual_audit_manifest_sha256": sha256(pose_audit),
    "sharp_manifest_sha256": sha256(sharp_manifest),
    "appearance_manifest_sha256": (
        sha256(appearance_manifest) if appearance_manifest != "none" else None
    ),
    "runner_sha256": sha256(runner),
    "fusion_script_sha256": sha256(fusion_script),
    "outputs": files,
    "numeric_gate_passed": quality["numeric_gate_passed"],
    "acceptance_status": quality["acceptance_status"],
    "provenance": "generated_fill_candidate",
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
    "stage": "MI300X_SHARP_metric_cross_view_fusion",
    "status": "done_candidate_pending_visual_review",
    "manifest_sha256": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(done_path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'MI300X SHARP metric fusion candidate complete: %s\n' "$run_dir"
