#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03}
parent_run=${ONELOOP_PARENT_TRAIN_RUN:?set ONELOOP_PARENT_TRAIN_RUN to the hash-verified formal train run}
dataset_manifest=${ONELOOP_OBJECT_DATASET_MANIFEST:?set ONELOOP_OBJECT_DATASET_MANIFEST to the hash-verified private manifest}
cameras=${ONELOOP_OBJECT_CAMERAS:?set ONELOOP_OBJECT_CAMERAS to the hash-verified camera JSON}
parent_manifest_sha=d27440b792f96f5f522a58e34ce091d97239595b1ffe86a1b5605130bd9ffb00
parent_metrics_sha=6c8cd274c80acbd67b00d721ae4d202b57847585cab0ee50ef6c891ad3fa36fd
dataset_manifest_sha=04900c1a3afc0f2f5e73ce0e34042e91a10c469a9c0b255e3157e7985e9b660e
cameras_sha=050891df1cfc5ef33070f7ab6becdd168267e5951143523519601f38963cbc26
parent_splat_sha=d95edcb66edd5fd3f6fe3fda4686dfe718ce867507a9c4db0fa6dde88a2cfcc5
vksplat_commit=e26c254938c81ff85998cd357a9e005e255d9b03
run_dir=${ONELOOP_RUN_DIR:?run through ops/run_job.sh so ONELOOP_RUN_DIR is recorded}
canonical_dir="$run_dir/canonical"
render_dir="$run_dir/render"
parent_manifest="$parent_run/manifest.json"
parent_metrics="$parent_run/metrics.json"
parent_ply="$parent_run/train/splat.ply"
parent_train_json="$parent_run/train/train.json"
canonical_ply="$canonical_dir/appearance_observed_canonical.ply"
provenance="$canonical_dir/provenance.json"

[[ ${ONELOOP_FORMAL_HOST:-} == radeon-c ]]
[[ ${ONELOOP_PARENT_CHECKPOINT:-} == "$parent_splat_sha" ]]
mkdir "$canonical_dir"
printf '%s  %s\n' \
  "$parent_manifest_sha" "$parent_manifest" \
  "$parent_metrics_sha" "$parent_metrics" \
  "$parent_splat_sha" "$parent_ply" \
  "$dataset_manifest_sha" "$dataset_manifest" \
  "$cameras_sha" "$cameras" \
  | sha256sum -c -
(cd "$parent_run/train" && sha256sum -c ../train_artifacts.sha256)

export PYTHONPATH="$repo_root/src:$repo_root"
"$python_bin" -m gaussian.canonicalize_vksplat_ply \
  --ply "$parent_ply" \
  --train-json "$parent_train_json" \
  --training-run-manifest "$parent_manifest" \
  --training-metrics "$parent_metrics" \
  --dataset-manifest "$dataset_manifest" \
  --output "$canonical_ply" \
  --output-provenance "$provenance" \
  --formal \
  --host-role radeon_c_gpu0_gfx1100_formal

started_ns=$(date +%s%N)
"$python_bin" -m gaussian.vksplat_render_ply \
  --ply "$canonical_ply" \
  --cameras "$cameras" \
  --source-provenance "$provenance" \
  --output "$render_dir" \
  --vksplat-root "$vksplat_root" \
  --vksplat-commit "$vksplat_commit" \
  --formal \
  --host-role radeon_c_gpu0_gfx1100_formal
finished_ns=$(date +%s%N)

"$python_bin" - "$run_dir/metrics.json" "$canonical_ply" "$provenance" \
  "$render_dir/render_manifest.json" "$started_ns" "$finished_ns" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from gaussian.vksplat_render_ply import read_3dgs_ply, sha256_file
output, ply_path, provenance_path, render_path = map(Path, sys.argv[1:5])
started_ns, finished_ns = map(int, sys.argv[5:7])
gaussians = read_3dgs_ply(ply_path)
provenance = json.loads(provenance_path.read_text())
render = json.loads(render_path.read_text())
extent = np.quantile(gaussians["xyz"], .995, axis=0) - np.quantile(
    gaussians["xyz"], .005, axis=0
)
checks = {
    "formal_render_true": render["formal"] is True,
    "formal_provenance_true": provenance["formal"] is True,
    "formal_host_role_exact": render["host_role"] == "radeon_c_gpu0_gfx1100_formal",
    "formal_parent_training_bound": provenance["training_lineage"]["training_formal"] is True,
    "secondary_accelerator_artifacts_excluded": provenance["training_lineage"]["secondary_accelerator_artifacts"] is False,
    "fixed_30000_gaussians": render["gaussian_count"] == 30000,
    "metric_extent_plausible": bool(np.all(extent > .04) and np.all(extent < .14)),
    "four_nonempty_renders": len(render["renders"]) == 4
    and all(0.0 < item["mean_transmittance"] < 1.0 for item in render["renders"]),
    "heldout_metrics_excluded": render["eligible_for_heldout_real_metrics"] is False,
}
report = {
    "schema_version": "radeon_oneloop.formal_object_vksplat_render.v1",
    "formal": True,
    "accepted": all(checks.values()),
    "checks": checks,
    "elapsed_render_s": (finished_ns - started_ns) / 1_000_000_000,
    "canonical_ply_sha256": sha256_file(ply_path),
    "canonical_provenance_sha256": sha256_file(provenance_path),
    "robust_extent_m_p005_p995": extent.tolist(),
    "renders": render["renders"],
    "vram_bytes": render["vram_bytes"],
    "peak_vram_bytes": render["peak_vram_bytes"],
    "heldout_quality_claim": False,
    "bitwise_checkpoint_determinism_claim": False,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not report["accepted"]:
    raise SystemExit("formal object VkSplat render acceptance gate failed")
PY

(cd "$run_dir" && find canonical render -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$run_dir/render_artifacts.sha256"
