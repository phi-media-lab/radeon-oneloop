#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03}
dataset=${ONELOOP_OBJECT_DATASET:?set ONELOOP_OBJECT_DATASET to the reviewed visual-hull dataset}
camera_stage=${ONELOOP_CAMERA_STAGE:?set ONELOOP_CAMERA_STAGE to the reviewed canonical camera stage}
run_root=${ONELOOP_RUN_ROOT:-/root/radeon-oneloop-runs/object_geometry_frozen_preflight}
steps=${ONELOOP_STEPS:-2000}
seed=${ONELOOP_SEED:-20260804}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_radeon_c_object_geometry_frozen_preflight"
run_dir="$run_root/$run_id"
train_dir="$run_dir/train"
canonical_dir="$run_dir/canonical"
render_dir="$run_dir/anchor_render"

[[ -x "$python_bin" ]]
[[ -f "$dataset/DONE" && -f "$dataset/dataset_manifest.json" ]]
[[ -f "$camera_stage/DONE" && -f "$camera_stage/cameras_observed.json" ]]
[[ -f "$vksplat_root/vksplat/simple_trainer.py" ]]
[[ "$steps" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$run_dir" "$canonical_dir"

write_hashes() {
  (cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
    -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
}
cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    write_hashes
    printf '{"status":"failed_nonformal_geometry_frozen_preflight","exit_code":%d}\n' "$status" \
      >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$repo_root/ops/assert_single_radeon.sh" gfx1100 "$python_bin" >"$run_dir/hardware.json"
"$python_bin" - \
  "$run_dir/manifest.json" "$run_id" "$steps" "$seed" "$dataset" "$camera_stage" \
  "$repo_root/gaussian/vksplat_train.py" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
path, run_id, steps, seed, dataset, camera_stage, trainer = sys.argv[1:]
sha = lambda value: hashlib.sha256(Path(value).read_bytes()).hexdigest()
value = {
    "schema_version": "radeon_oneloop.object_geometry_frozen_preflight_run.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "radeon_c_gpu0_gfx1100_nonformal",
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "steps": int(steps),
    "seed": int(seed),
    "dataset_manifest_sha256": sha(Path(dataset) / "dataset_manifest.json"),
    "cameras_sha256": sha(Path(camera_stage) / "cameras_observed.json"),
    "trainer_sha256": sha(trainer),
    "optimization": "freeze_visual_hull_means_scales_quaternions_fit_dc_and_opacity",
    "observed_only_training": True,
    "secondary_accelerator_artifacts": False,
    "heldout_real_metrics": False,
    "physical_output": False,
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"
export XDG_RUNTIME_DIR="$xdg_runtime"
export PYTHONPATH="$repo_root/src:$repo_root"

ONELOOP_RUN_DIR="$run_dir" timeout --signal=TERM --kill-after=30 1800 \
  "$python_bin" -m gaussian.vksplat_train \
  --source "$vksplat_root" \
  --dataset "$dataset" \
  --output "$train_dir" \
  --image-dir images \
  --mask-dir masks \
  --sparse-dir sparse/0 \
  --min-images 5 \
  --steps "$steps" \
  --seed "$seed" \
  --eval-interval 8 \
  --strategy default \
  --freeze-higher-sh \
  --freeze-geometry \
  --disable-refinement \
  --host-role radeon_c_gpu0_gfx1100_nonformal \
  >"$run_dir/train_stdout.log" 2>"$run_dir/train_stderr.log"

"$python_bin" -m gaussian.canonicalize_vksplat_ply \
  --ply "$train_dir/splat.ply" \
  --train-json "$train_dir/train.json" \
  --output "$canonical_dir/appearance_observed_canonical.ply" \
  --output-provenance "$canonical_dir/provenance.json" \
  --host-role radeon_c_gpu0_gfx1100_nonformal \
  >"$run_dir/canonicalize.log" 2>&1
cp "$camera_stage/cameras_observed.json" "$canonical_dir/cameras_observed.json"

"$python_bin" -m gaussian.vksplat_render_ply \
  --ply "$canonical_dir/appearance_observed_canonical.ply" \
  --cameras "$canonical_dir/cameras_observed.json" \
  --source-provenance "$canonical_dir/provenance.json" \
  --output "$render_dir" \
  --vksplat-root "$vksplat_root" \
  --vksplat-commit e26c254938c81ff85998cd357a9e005e255d9b03 \
  --host-role radeon_c_gpu0_gfx1100_nonformal \
  >"$run_dir/render.log" 2>&1

"$python_bin" - "$run_dir/metrics.json" "$train_dir/oneloop_metrics.json" \
  "$canonical_dir/provenance.json" "$render_dir/render_manifest.json" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from gaussian.vksplat_render_ply import read_3dgs_ply
output_path, train_path, provenance_path, render_path = map(Path, sys.argv[1:])
train = json.loads(train_path.read_text())
provenance = json.loads(provenance_path.read_text())
render = json.loads(render_path.read_text())
profile = train["optimization_profile"]
gaussians = read_3dgs_ply(Path(provenance_path).parent / "appearance_observed_canonical.ply")
checks = {
    "nonformal": train["formal"] is False and provenance["formal"] is False,
    "observed_only": provenance["observed_only_training"] is True,
    "fixed_30000_splats": train["num_splats"] == 30000 == provenance["gaussian_count"],
    "geometry_frozen": profile["freeze_geometry"] is True,
    "center_rates_negligible": (
        0.0 < profile["means_lr"] <= 1.0e-12
        and 0.0 < profile["means_lr_final"] <= 1.0e-12
    ),
    "shape_rates_zero": profile["scales_lr"] == 0.0 and profile["quats_lr"] == 0.0,
    "all_gaussian_fields_finite": all(np.isfinite(gaussians[name]).all() for name in (
        "xyz", "scales", "opacities", "rotations", "sh"
    )),
    "anchor_renders_complete": len(render["renders"]) == 4,
    "no_heldout_claim": provenance["eligible_for_heldout_real_metrics"] is False,
}
report = {
    "schema_version": "radeon_oneloop.object_geometry_frozen_preflight.v1",
    "formal": False,
    "accepted": all(checks.values()),
    "checks": checks,
    "train": train,
    "canonical_ply_sha256": provenance["output_ply_sha256"],
    "canonical_provenance": provenance,
    "anchor_render": render,
    "physical_output": False,
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"accepted": report["accepted"], "checks": checks}, indent=2, sort_keys=True))
if not report["accepted"]:
    raise RuntimeError("geometry-frozen preflight failed")
PY

write_hashes
printf '{"status":"done_nonformal_pending_continuous_orbit_review"}\n' >"$run_dir/DONE"
trap - EXIT INT TERM
printf 'Radeon-c geometry-frozen object preflight passed: %s\n' "$run_dir"
