#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03}
dataset=${ONELOOP_OBJECT_DATASET:?set ONELOOP_OBJECT_DATASET to the hash-verified private dataset root}
dataset_manifest_sha=04900c1a3afc0f2f5e73ce0e34042e91a10c469a9c0b255e3157e7985e9b660e
dataset_hash=682b65e97653ffe08e469496bb0554f349aeff103ddf8e57f1e4857f8c04534e
seed=${ONELOOP_SEED:-20260804}
steps=2000
run_dir=${ONELOOP_RUN_DIR:?run through ops/run_job.sh so ONELOOP_RUN_DIR is recorded}
train_dir="$run_dir/train"

[[ ${ONELOOP_FORMAL_HOST:-} == radeon-c ]]
[[ -x "$python_bin" ]]
[[ -f "$vksplat_root/vksplat/simple_trainer.py" ]]
[[ -f "$dataset/DONE" && -f "$dataset/dataset_manifest.json" ]]
printf '%s  %s\n' "$dataset_manifest_sha" "$dataset/dataset_manifest.json" \
  | sha256sum -c -

export PYTHONPATH="$repo_root/src:$repo_root"
"$python_bin" - "$dataset" "$dataset_hash" <<'PY'
import sys
from pathlib import Path
from gaussian.vksplat_train import validate_dataset
root, expected = Path(sys.argv[1]), sys.argv[2]
report = validate_dataset(root, "images", "sparse/0", min_images=5, mask_dir="masks")
if report["dataset_hash"] != expected:
    raise SystemExit(f"dataset hash mismatch: {report['dataset_hash']} != {expected}")
PY

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
  --formal \
  --host-role radeon_c_gpu0_gfx1100_formal

"$python_bin" - "$run_dir/metrics.json" "$train_dir/train.json" "$dataset_hash" "$steps" "$seed" <<'PY'
import json, sys
from pathlib import Path
metrics_path, train_path, dataset_hash, steps, seed = sys.argv[1:]
metrics_path, train_path = Path(metrics_path), Path(train_path)
metrics = json.loads(metrics_path.read_text())
train = json.loads(train_path.read_text())
train_names = sorted(Path(item["image_path"]).name for item in train["train_images"])
val_names = [Path(item["image_path"]).name for item in train["val_images"]]
checks = {
    "formal_flag_true": metrics["formal"] is True,
    "formal_host_role_exact": metrics["host_role"] == "radeon_c_gpu0_gfx1100_formal",
    "dataset_hash_exact": metrics["dataset"]["dataset_hash"] == dataset_hash,
    "steps_exact": metrics["steps"] == int(steps),
    "seed_exact": metrics["seed"] == int(seed),
    "fixed_30000_splats": metrics["num_splats"] == 30000,
    "four_unique_observed_train_views": train_names == [
        "anchor_front.png", "anchor_left.png", "anchor_rear.png", "anchor_right.png"
    ],
    "front_duplicate_only_validation_view": val_names == ["000_eval_probe_anchor_front.png"],
    "no_heldout_quality_claim": metrics["evaluation"] is None,
}
metrics["formal_acceptance"] = {
    "accepted": all(checks.values()),
    "checks": checks,
    "generated_fill_enabled": False,
    "secondary_accelerator_artifacts": False,
    "bitwise_checkpoint_determinism_claim": False,
    "train_images": train_names,
    "validation_images": val_names,
}
metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not metrics["formal_acceptance"]["accepted"]:
    raise SystemExit("formal object VkSplat acceptance gate failed")
PY

(cd "$train_dir" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$run_dir/train_artifacts.sha256"
