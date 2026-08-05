#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/home/amd/sim/vksplat}
: "${ONELOOP_SEVA_FULL_GEOMETRY_DATASET:?set ONELOOP_SEVA_FULL_GEOMETRY_DATASET}"
dataset=$ONELOOP_SEVA_FULL_GEOMETRY_DATASET
run_root=${ONELOOP_SEVA_FULL_GEOMETRY_RUN_ROOT:-/home/amd/radeon-oneloop-runs/seva_full_geometry}
steps=${ONELOOP_SEVA_FULL_GEOMETRY_STEPS:-8000}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_seva_full_geometry"
run_dir="$run_root/$run_id"
train_dir="$run_dir/train"
asset_dir="$run_dir/asset"

[[ -x "$python_bin" ]]
[[ -d "$vksplat_root/vksplat/shader" ]]
[[ -f "$dataset/DONE" ]]
[[ -f "$dataset/dataset_manifest.json" ]]
[[ "$steps" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$train_dir" "$asset_dir"

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    printf '{"status":"failed","exit_code":%d}\n' "$status" >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

export PYTHONPATH="$repo_root/src:$repo_root:$vksplat_root/build"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"
export XDG_RUNTIME_DIR="$xdg_runtime"

"$python_bin" -m gaussian.provenance_quarantine \
  --check-json "$dataset/dataset_manifest.json"

image_count=$(find "$dataset/images" -maxdepth 1 -type f -name '*.png' | wc -l)
"$python_bin" - "$dataset" "$image_count" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
image_count = int(sys.argv[2])
manifest_path = root / "dataset_manifest.json"
manifest = json.loads(manifest_path.read_text())
done = json.loads((root / "DONE").read_text())
if done.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
    raise SystemExit("dataset DONE does not bind manifest")
if manifest.get("schema_version") != "radeon_oneloop.seva_full_geometry_colmap_dataset.v1":
    raise SystemExit("unexpected full-geometry dataset schema")
if manifest.get("formal") is not False or manifest.get("eligible_for_collision_geometry") is not False:
    raise SystemExit("full-geometry dataset weakened its evidence boundary")
if image_count != manifest["sampling"]["training_instances_total"] + 1:
    raise SystemExit("dataset image count differs from its sampling contract")
initial = manifest.get("initial_points", {})
if initial.get("source") != "49_view_generated_SEVA_support_visual_hull":
    raise SystemExit("dataset lacks the generated multi-view hull")
if initial.get("generated_geometry_prior") is not True:
    raise SystemExit("generated geometry must remain explicit")
if initial.get("old_surface_points_inherited") is not False:
    raise SystemExit("old surface points must not enter the candidate")
if manifest.get("lineage", {}).get("inherited_mesh_or_procedural_surface") is not None:
    raise SystemExit("a mesh or procedural surface entered the candidate")
profile = manifest.get("required_training_profile", {})
if profile.get("freeze_geometry") is not False or profile.get("disable_refinement") is not False:
    raise SystemExit("full-geometry training must optimize and refine geometry")
PY

vulkaninfo --summary >"$run_dir/vulkaninfo_summary.txt" 2>&1
if ! grep -q 'vendorID.*0x1002' "$run_dir/vulkaninfo_summary.txt" \
  || ! grep -q 'RADV GFX1150' "$run_dir/vulkaninfo_summary.txt"; then
  printf '%s\n' 'expected the amd APU RADV GFX1150 device' >&2
  exit 69
fi

command=(
  "$python_bin" -m gaussian.vksplat_train
  --source "$vksplat_root"
  --dataset "$dataset"
  --output "$train_dir"
  --image-dir images
  --mask-dir masks
  --sparse-dir sparse/0
  --min-images "$image_count"
  --steps "$steps"
  --eval-interval "$image_count"
  --strategy default
  --freeze-higher-sh
  --init-scale 0.5
  --scale-reg 0.01
  --opacity-reg 0.01
  --scales-lr 0.002
  --quats-lr 0.0005
  --grow-grad2d 0.0003
  --grow-scale3d 0.005
  --grow-scale2d 0.03
  --prune-scale3d 0.025
  --prune-scale2d 0.08
  --refine-stop-iter 6000
  --seed 20260805
  --host-role amd_apu_gfx1150_nonformal
)
printf '%q ' "${command[@]}" >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

timeout --signal=TERM --kill-after=30 1800 \
  "${command[@]}" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

"$python_bin" - "$run_dir/train_manifest.json" "$run_id" "$dataset" \
  "$train_dir/oneloop_metrics.json" "$vksplat_root" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

output, run_id, dataset, metrics_path, vksplat_root = map(Path, sys.argv[1:])
metrics = json.loads(metrics_path.read_text())
dataset_manifest = dataset / "dataset_manifest.json"
commit = subprocess.run(
    ["git", "-C", str(vksplat_root), "rev-parse", "HEAD"],
    check=False, capture_output=True, text=True,
).stdout.strip() or "unknown"
value = {
    "schema_version": "radeon_oneloop.seva_full_geometry_training.v1",
    "run_id": run_id.name,
    "formal": False,
    "host": "amd",
    "host_role": "amd_apu_gfx1150_nonformal",
    "role": "SEVA_49_view_generated_geometry_hypothesis_variable_3DGS",
    "review_status": "pending_human_full_orbit_geometry_audit",
    "collision_eligible": False,
    "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
    "dataset_hash": metrics["dataset"]["dataset_hash"],
    "vksplat_commit": commit,
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

"$python_bin" -m gaussian.canonicalize_vksplat_ply \
  --ply "$train_dir/splat.ply" \
  --train-json "$train_dir/train.json" \
  --output "$asset_dir/appearance_full_geometry_canonical.ply" \
  --output-provenance "$asset_dir/provenance.json" \
  --training-run-manifest "$run_dir/train_manifest.json" \
  --training-metrics "$train_dir/oneloop_metrics.json" \
  --dataset-manifest "$dataset/dataset_manifest.json" \
  --host-role amd_apu_gfx1150_nonformal \
  --provenance-class generated_full_geometry_candidate \
  >"$run_dir/canonicalize.log"

"$python_bin" -m gaussian.colmap_cardinal_camera_export \
  --dataset "$dataset" \
  --output "$asset_dir/cameras_full_geometry.json" \
  --mode full_geometry_training_views \
  >"$run_dir/cameras.log"

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
printf '{"status":"done_candidate_pending_human_geometry_audit","asset_dir":"%s"}\n' \
  "$asset_dir" >"$run_dir/DONE"
trap - EXIT INT TERM
printf '%s\n' "$asset_dir"
