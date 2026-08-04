#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03}
parent_run=${ONELOOP_PARENT_TRAIN_RUN:?set ONELOOP_PARENT_TRAIN_RUN to the hash-verified formal train run}
dataset_manifest=${ONELOOP_OBJECT_DATASET_MANIFEST:?set ONELOOP_OBJECT_DATASET_MANIFEST to the hash-verified private manifest}
cameras=${ONELOOP_OBJECT_CAMERAS:?set ONELOOP_OBJECT_CAMERAS to the hash-verified camera JSON}
parent_manifest_sha=45a96959f3f25658804ba2d2be335b785881d89e0767bb7f56bb4f60cbcb9c1d
parent_metrics_sha=5ae68080a52a3f0aadf3e7967e3ed0f74f6856c0c71953c082410e79583896ec
dataset_manifest_sha=04900c1a3afc0f2f5e73ce0e34042e91a10c469a9c0b255e3157e7985e9b660e
cameras_sha=050891df1cfc5ef33070f7ab6becdd168267e5951143523519601f38963cbc26
parent_splat_sha=e9be3a2df4c1ca7fcfddc86deee4c366a2f941f66a881e41d13367c329aff378
vksplat_commit=e26c254938c81ff85998cd357a9e005e255d9b03
run_dir=${ONELOOP_RUN_DIR:?run through ops/run_job.sh so ONELOOP_RUN_DIR is recorded}
canonical_dir="$run_dir/canonical"
render_dir="$run_dir/render"
parent_manifest="$parent_run/manifest.json"
parent_metrics="$parent_run/metrics.json"
parent_ply="$parent_run/train/splat.ply"
parent_train_json="$parent_run/train/train.json"
parent_config="$parent_run/config.yaml"
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
"$python_bin" - "$parent_metrics" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))
profile = metrics["optimization_profile"]
acceptance = metrics["formal_acceptance"]
checks = {
    "formal_parent_accepted": acceptance["accepted"] is True,
    "geometry_frozen": profile["freeze_geometry"] is True,
    "center_rates_negligible": 0.0 < profile["means_lr"] <= 1.0e-12
    and 0.0 < profile["means_lr_final"] <= 1.0e-12,
    "shape_rates_zero": profile["scales_lr"] == 0.0 and profile["quats_lr"] == 0.0,
    "generated_fill_disabled": acceptance["generated_fill_enabled"] is False,
    "secondary_accelerator_artifacts_excluded": acceptance["secondary_accelerator_artifacts"] is False,
}
if not all(checks.values()):
    raise SystemExit(f"formal parent geometry-frozen gate failed: {checks}")
PY

"$python_bin" -m gaussian.canonicalize_vksplat_ply \
  --ply "$parent_ply" \
  --train-json "$parent_train_json" \
  --training-run-manifest "$parent_manifest" \
  --training-metrics "$parent_metrics" \
  --training-config "$parent_config" \
  --vksplat-commit "$vksplat_commit" \
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
extent = np.quantile(gaussians["xyz"], .9999, axis=0) - np.quantile(
    gaussians["xyz"], .0001, axis=0
)
checks = {
    "formal_render_true": render["formal"] is True,
    "formal_provenance_true": provenance["formal"] is True,
    "observed_only_training": provenance["observed_only_training"] is True,
    "formal_host_role_exact": render["host_role"] == "radeon_c_gpu0_gfx1100_formal",
    "formal_parent_training_bound": provenance["training_lineage"]["training_formal"] is True,
    "parent_geometry_frozen_checkpoint_exact": provenance["training_lineage"]["input_ply_sha256"] == "e9be3a2df4c1ca7fcfddc86deee4c366a2f941f66a881e41d13367c329aff378",
    "secondary_accelerator_artifacts_excluded": provenance["training_lineage"]["secondary_accelerator_artifacts"] is False,
    "fixed_30000_gaussians": render["gaussian_count"] == 30000,
    "metric_extent_plausible": bool(np.all(extent > .04) and np.all(extent < .14)),
    "four_nonempty_renders": len(render["renders"]) == 4
    and all(0.0 < item["mean_transmittance"] < 1.0 for item in render["renders"]),
    "heldout_metrics_excluded": render["eligible_for_heldout_real_metrics"] is False,
}
report = {
    "schema_version": "radeon_oneloop.formal_object_geometry_frozen_render.v1",
    "formal": True,
    "accepted": all(checks.values()),
    "checks": checks,
    "elapsed_render_s": (finished_ns - started_ns) / 1_000_000_000,
    "canonical_ply_sha256": sha256_file(ply_path),
    "canonical_provenance_sha256": sha256_file(provenance_path),
    "robust_extent_m_p0001_p9999": extent.tolist(),
    "renders": render["renders"],
    "vram_bytes": render["vram_bytes"],
    "peak_vram_bytes": render["peak_vram_bytes"],
    "generated_fill_enabled": False,
    "heldout_quality_claim": False,
    "bitwise_checkpoint_determinism_claim": False,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not report["accepted"]:
    raise SystemExit("formal geometry-frozen object render acceptance gate failed")
PY

(cd "$run_dir" && find canonical render -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$run_dir/render_artifacts.sha256"
