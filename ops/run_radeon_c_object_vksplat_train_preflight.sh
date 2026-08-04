#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03}
dataset=${ONELOOP_OBJECT_DATASET:-/root/radeon-oneloop-data/object_assets/graffiti_mickey_asset_v1/formal_inputs/manual_ring_visual_hull_r160_v1}
run_root=${ONELOOP_RUN_ROOT:-/root/radeon-oneloop-runs/object_vksplat_train_preflight}
steps=${ONELOOP_STEPS:-2000}
seed=${ONELOOP_SEED:-20260804}
dataset_manifest_sha=04900c1a3afc0f2f5e73ce0e34042e91a10c469a9c0b255e3157e7985e9b660e
dataset_hash=682b65e97653ffe08e469496bb0554f349aeff103ddf8e57f1e4857f8c04534e
vksplat_commit=e26c254938c81ff85998cd357a9e005e255d9b03
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_radeon_c_object_vksplat_train_preflight"
run_dir="$run_root/$run_id"
train_dir="$run_dir/train"
trainer="$repo_root/gaussian/vksplat_train.py"

[[ -x "$python_bin" ]]
[[ -f "$trainer" ]]
[[ -f "$dataset/DONE" && -f "$dataset/dataset_manifest.json" ]]
[[ -f "$vksplat_root/vksplat/simple_trainer.py" ]]
[[ "$steps" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$run_dir"

write_hashes() {
  (cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
    -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
}
cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    write_hashes
    "$python_bin" - "$run_dir/FAILED.tmp" "$status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status = sys.argv[1:]
value = {
    "stage": "radeon_c_object_vksplat_train_preflight",
    "status": "failed_nonformal_preflight",
    "exit_code": int(status),
    "failed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
    mv "$run_dir/FAILED.tmp" "$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s  %s\n' "$dataset_manifest_sha" "$dataset/dataset_manifest.json" \
  | sha256sum -c - >"$run_dir/dataset_manifest_hash_check.log"
"$repo_root/ops/assert_single_radeon.sh" gfx1100 "$python_bin" \
  >"$run_dir/hardware.json"
vulkaninfo --summary >"$run_dir/vulkaninfo_summary.txt" 2>&1
grep -q 'vendorID.*0x1002' "$run_dir/vulkaninfo_summary.txt"
grep -q 'RADV NAVI31' "$run_dir/vulkaninfo_summary.txt"

export PYTHONPATH="$repo_root/src:$repo_root"
"$python_bin" - "$dataset" "$dataset_hash" >"$run_dir/dataset_validation.json" <<'PY'
import json, sys
from pathlib import Path
from gaussian.vksplat_train import validate_dataset
root, expected = Path(sys.argv[1]), sys.argv[2]
report = validate_dataset(root, "images", "sparse/0", min_images=5, mask_dir="masks")
if report["dataset_hash"] != expected:
    raise SystemExit(f"dataset hash mismatch: {report['dataset_hash']} != {expected}")
print(json.dumps(report, indent=2, sort_keys=True))
PY

"$python_bin" - \
  "$run_dir/manifest.json" "$run_id" "$steps" "$seed" "$dataset" "$dataset_manifest_sha" \
  "$dataset_hash" "$trainer" "$vksplat_root" "$vksplat_commit" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
(
    path, run_id, steps, seed, dataset, dataset_manifest_sha, dataset_hash,
    trainer, vksplat_root, vksplat_commit,
) = sys.argv[1:]
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
value = {
    "schema_version": "radeon_oneloop.object_vksplat_train_preflight_run.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "radeon_c_gpu0_gfx1100_nonformal",
    "gpu": "GPU0",
    "gfx_target": "gfx1100",
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "mode": "sparse_static",
    "steps": int(steps),
    "seed": int(seed),
    "dataset": dataset,
    "dataset_manifest_sha256": dataset_manifest_sha,
    "dataset_hash": dataset_hash,
    "trainer_sha256": sha256(trainer),
    "vksplat_root": vksplat_root,
    "vksplat_commit": vksplat_commit,
    "vksplat_checkout": "clean_pinned_archive_with_runtime_shader_path_adapter",
    "generated_fill_enabled": False,
    "heldout_quality_claim": False,
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

started_ns=$(date +%s%N)
XDG_RUNTIME_DIR="$xdg_runtime" ONELOOP_RUN_DIR="$run_dir" \
  timeout --signal=TERM --kill-after=30 1800 \
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
  --disable-refinement \
  --host-role radeon_c_gpu0_gfx1100_nonformal \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
finished_ns=$(date +%s%N)

"$python_bin" - \
  "$run_dir/metrics.json" "$run_dir/train/oneloop_metrics.json" \
  "$run_dir/train/train.json" "$dataset_hash" "$dataset_manifest_sha" \
  "$steps" "$seed" "$started_ns" "$finished_ns" <<'PY'
import json, sys
from pathlib import Path
(
    output_path, metrics_path, train_path, expected_dataset_hash,
    dataset_manifest_sha, steps, seed, started_ns, finished_ns,
) = sys.argv[1:]
metrics = json.loads(Path(metrics_path).read_text())
train = json.loads(Path(train_path).read_text())
train_names = sorted(Path(item["image_path"]).name for item in train["train_images"])
val_names = [Path(item["image_path"]).name for item in train["val_images"]]
checks = {
    "formal_flag_false": metrics["formal"] is False,
    "formal_host_not_claimed": metrics["host_role"] == "radeon_c_gpu0_gfx1100_nonformal",
    "dataset_hash_exact": metrics["dataset"]["dataset_hash"] == expected_dataset_hash,
    "dataset_manifest_hash_exact": len(dataset_manifest_sha) == 64,
    "steps_exact": metrics["steps"] == int(steps),
    "seed_exact": metrics["seed"] == int(seed),
    "fixed_30000_splats": metrics["num_splats"] == 30000,
    "four_unique_observed_train_views": train_names == [
        "anchor_front.png", "anchor_left.png", "anchor_rear.png", "anchor_right.png"
    ],
    "front_duplicate_only_validation_view": val_names == ["000_eval_probe_anchor_front.png"],
    "no_heldout_quality_claim": metrics["evaluation"] is None,
}
report = {
    "schema_version": "radeon_oneloop.radeon_c_object_vksplat_train_preflight.v1",
    "formal": False,
    "accepted": all(checks.values()),
    "checks": checks,
    "elapsed_wall_s": (int(finished_ns) - int(started_ns)) / 1_000_000_000,
    "training_elapsed_s": metrics["elapsed_seconds"],
    "num_splats": metrics["num_splats"],
    "vram_bytes": metrics["vram_bytes"],
    "peak_vram_bytes": metrics["peak_vram_bytes"],
    "dataset_hash": metrics["dataset"]["dataset_hash"],
    "dataset_manifest_sha256": dataset_manifest_sha,
    "train_images": train_names,
    "validation_images": val_names,
    "splat_sha256": metrics["artifacts"]["splat.ply"],
    "eligible_for_formal_rerun": all(checks.values()),
}
Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["accepted"]:
    raise RuntimeError("Radeon-c object VkSplat training preflight failed")
PY

write_hashes
(cd "$run_dir" && sha256sum -c hashes.sha256 >/dev/null)
"$python_bin" - "$run_dir/DONE.tmp" "$run_dir/manifest.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
path, manifest = map(Path, sys.argv[1:])
value = {
    "stage": "radeon_c_object_vksplat_train_preflight",
    "status": "done_nonformal_formal_rerun_candidate",
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mv "$run_dir/DONE.tmp" "$run_dir/DONE"
trap - EXIT INT TERM
printf 'Radeon-c object VkSplat training preflight passed: %s\n' "$run_dir"
