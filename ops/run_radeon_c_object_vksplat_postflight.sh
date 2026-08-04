#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03}
training_run=${ONELOOP_TRAINING_RUN:-/root/radeon-oneloop-runs/object_vksplat_train_preflight/20260804T094622Z_1952396_radeon_c_object_vksplat_train_preflight}
dataset=${ONELOOP_OBJECT_DATASET:-/root/radeon-oneloop-data/object_assets/graffiti_mickey_asset_v1/formal_inputs/manual_ring_visual_hull_r160_v1}
camera_stage=${ONELOOP_CAMERA_STAGE:-/root/radeon-oneloop-data/object_assets/graffiti_mickey_asset_v1/formal_inputs/manual_ring_pose_v1}
run_root=${ONELOOP_RUN_ROOT:-/root/radeon-oneloop-runs/object_vksplat_postflight}
vksplat_commit=e26c254938c81ff85998cd357a9e005e255d9b03
input_ply_sha=${ONELOOP_INPUT_PLY_SHA:-aac51a06bcd257d44c1432759e827891c94f3b4e35100864defbe3353aaccafd}
train_json_sha=${ONELOOP_TRAIN_JSON_SHA:-fb258b09b07c924acd39075642a8fb1f5a31fbc7753415306bc4a11fd598555c}
training_manifest_sha=${ONELOOP_TRAINING_MANIFEST_SHA:-f71e8a9272275d2e47429010fcfb7fca44d41efd63ccc44bc6560130b6562252}
dataset_manifest_sha=04900c1a3afc0f2f5e73ce0e34042e91a10c469a9c0b255e3157e7985e9b660e
cameras_sha=050891df1cfc5ef33070f7ab6becdd168267e5951143523519601f38963cbc26
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_radeon_c_object_vksplat_postflight"
run_dir="$run_root/$run_id"
canonical_dir="$run_dir/canonical"
render_dir="$run_dir/render"
input_ply="$training_run/train/splat.ply"
train_json="$training_run/train/train.json"
training_manifest="$training_run/manifest.json"
dataset_manifest="$dataset/dataset_manifest.json"
cameras="$camera_stage/cameras_observed.json"
canonical_ply="$canonical_dir/appearance_observed_canonical.ply"
provenance="$canonical_dir/provenance.json"

[[ -x "$python_bin" ]]
[[ -f "$vksplat_root/vksplat/simple_trainer.py" ]]
[[ -f "$training_run/DONE" && -f "$dataset/DONE" && -f "$camera_stage/DONE" ]]
mkdir -p "$canonical_dir"

write_hashes() {
  (cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
    -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
}
cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    write_hashes
    printf '{"status":"failed_nonformal_postflight","exit_code":%d}\n' "$status" \
      >"$run_dir/FAILED.tmp"
    mv "$run_dir/FAILED.tmp" "$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s  %s\n' \
  "$input_ply_sha" "$input_ply" \
  "$train_json_sha" "$train_json" \
  "$training_manifest_sha" "$training_manifest" \
  "$dataset_manifest_sha" "$dataset_manifest" \
  "$cameras_sha" "$cameras" \
  | sha256sum -c - >"$run_dir/input_hash_check.log"
"$repo_root/ops/assert_single_radeon.sh" gfx1100 "$python_bin" \
  >"$run_dir/hardware.json"
vulkaninfo --summary >"$run_dir/vulkaninfo_summary.txt" 2>&1
grep -q 'vendorID.*0x1002' "$run_dir/vulkaninfo_summary.txt"
grep -q 'RADV NAVI31' "$run_dir/vulkaninfo_summary.txt"

"$python_bin" - \
  "$run_dir/manifest.json" "$run_id" "$training_run" "$training_manifest_sha" \
  "$dataset_manifest_sha" "$cameras_sha" "$repo_root/gaussian/canonicalize_vksplat_ply.py" \
  "$repo_root/gaussian/vksplat_render_ply.py" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
(
    path, run_id, training_run, training_manifest_sha, dataset_manifest_sha,
    cameras_sha, canonicalizer, renderer,
) = sys.argv[1:]
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
value = {
    "schema_version": "radeon_oneloop.object_vksplat_postflight_run.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "radeon_c_gpu0_gfx1100_nonformal",
    "gpu": "GPU0",
    "gfx_target": "gfx1100",
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "training_run": training_run,
    "training_manifest_sha256": training_manifest_sha,
    "dataset_manifest_sha256": dataset_manifest_sha,
    "cameras_sha256": cameras_sha,
    "canonicalizer_sha256": sha256(canonicalizer),
    "renderer_sha256": sha256(renderer),
    "vksplat_commit": "e26c254938c81ff85998cd357a9e005e255d9b03",
    "heldout_quality_claim": False,
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

export PYTHONPATH="$repo_root/src:$repo_root"
"$python_bin" -m gaussian.canonicalize_vksplat_ply \
  --ply "$input_ply" \
  --train-json "$train_json" \
  --training-run-manifest "$training_manifest" \
  --dataset-manifest "$dataset_manifest" \
  --output "$canonical_ply" \
  --output-provenance "$provenance" \
  >"$run_dir/canonicalize.log" 2>&1

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"
started_ns=$(date +%s%N)
XDG_RUNTIME_DIR="$xdg_runtime" timeout --signal=TERM --kill-after=15 300 \
  "$python_bin" -m gaussian.vksplat_render_ply \
  --ply "$canonical_ply" \
  --cameras "$cameras" \
  --source-provenance "$provenance" \
  --output "$render_dir" \
  --vksplat-root "$vksplat_root" \
  --vksplat-commit "$vksplat_commit" \
  >"$run_dir/render.log" 2>&1
finished_ns=$(date +%s%N)

"$python_bin" - \
  "$run_dir/metrics.json" "$canonical_ply" "$provenance" \
  "$render_dir/render_manifest.json" "$training_manifest_sha" \
  "$dataset_manifest_sha" "$started_ns" "$finished_ns" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from gaussian.vksplat_render_ply import read_3dgs_ply, sha256_file
(
    output_path, ply_path, provenance_path, render_path,
    training_manifest_sha, dataset_manifest_sha, started_ns, finished_ns,
) = sys.argv[1:]
ply_path, provenance_path, render_path = map(Path, (ply_path, provenance_path, render_path))
gaussians = read_3dgs_ply(ply_path)
provenance = json.loads(provenance_path.read_text())
render = json.loads(render_path.read_text())
robust_extent = np.quantile(gaussians["xyz"], 0.995, axis=0) - np.quantile(
    gaussians["xyz"], 0.005, axis=0
)
lineage = provenance["training_lineage"]
checks = {
    "formal_flag_false": render["formal"] is False,
    "observed_core_provenance": provenance["provenance_class"] == "observed_core_candidate",
    "training_manifest_bound": lineage["training_run_manifest_sha256"] == training_manifest_sha,
    "dataset_manifest_bound": lineage["dataset_manifest_sha256"] == dataset_manifest_sha,
    "secondary_accelerator_artifacts_excluded": lineage["secondary_accelerator_artifacts"] is False,
    "canonical_ply_hash_bound": provenance["output_ply_sha256"] == sha256_file(ply_path),
    "fixed_30000_gaussians": render["gaussian_count"] == 30000 == len(gaussians["xyz"]),
    "metric_extent_plausible": bool(np.all(robust_extent > 0.04) and np.all(robust_extent < 0.14)),
    "four_nonempty_renders": len(render["renders"]) == 4
    and all(0.0 < item["mean_transmittance"] < 1.0 for item in render["renders"]),
    "observed_qa_purpose": render["purpose"] == "nonformal observed-core visual QA",
    "no_formal_metric_eligibility": render["eligible_for_formal_metrics"] is False,
}
report = {
    "schema_version": "radeon_oneloop.radeon_c_object_vksplat_postflight.v1",
    "formal": False,
    "accepted": all(checks.values()),
    "checks": checks,
    "elapsed_render_s": (int(finished_ns) - int(started_ns)) / 1_000_000_000,
    "canonical_ply_sha256": sha256_file(ply_path),
    "canonical_provenance_sha256": sha256_file(provenance_path),
    "robust_extent_m_p005_p995": robust_extent.tolist(),
    "renders": render["renders"],
    "vram_bytes": render["vram_bytes"],
    "peak_vram_bytes": render["peak_vram_bytes"],
    "eligible_for_visual_review": all(checks.values()),
}
Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["accepted"]:
    raise RuntimeError("Radeon-c object VkSplat postflight failed")
PY

write_hashes
(cd "$run_dir" && sha256sum -c hashes.sha256 >/dev/null)
printf '{"status":"done_nonformal_pending_visual_review"}\n' >"$run_dir/DONE.tmp"
mv "$run_dir/DONE.tmp" "$run_dir/DONE"
trap - EXIT INT TERM
printf 'Radeon-c object VkSplat postflight passed: %s\n' "$run_dir"
