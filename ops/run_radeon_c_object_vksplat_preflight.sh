#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
vksplat_root=${ONELOOP_VKSPLAT_ROOT:-/root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03}
observed_root=${ONELOOP_OBSERVED_CORE_ROOT:-/root/radeon-oneloop-data/object_assets/graffiti_mickey_asset_v1/observed_core/canonicalized_vksplat_20260804}
run_root=${ONELOOP_RUN_ROOT:-/root/radeon-oneloop-runs/object_vksplat_preflight}
config="$repo_root/configs/gaussian_object_formal_candidate.yaml"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_radeon_c_object_vksplat_preflight"
run_dir="$run_root/$run_id"
ply="$observed_root/appearance_observed_canonical.ply"
cameras="$observed_root/cameras_observed.json"
provenance="$observed_root/provenance.json"

[[ -x "$python_bin" ]]
[[ -f "$config" ]]
[[ -f "$vksplat_root/vksplat/simple_trainer.py" ]]
[[ -f "$ply" && -f "$cameras" && -f "$provenance" ]]
mkdir -p "$run_dir"

write_hashes() {
  (cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
    -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
}
cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    write_hashes
    printf '{"status":"failed","exit_code":%d}\n' "$status" >"$run_dir/FAILED"
  fi
  exit "$status"
}
trap cleanup EXIT

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

printf '%s  %s\n' \
  ad30184dd9b39fdc211404538e70c0544d2c418b1af2003ae6f07353c5471183 "$ply" \
  4d03ed63829d786325adf29e4a6c1fb2542d0a090ba3b66ea1e8b099d994cb00 "$cameras" \
  9e8e70efb8c9d5d98e9e792f9f0c06836eb1127cdfd1e8d8dceda9e728441bf9 "$provenance" \
  | sha256sum -c - >"$run_dir/asset_hash_check.log"

"$repo_root/ops/assert_single_radeon.sh" gfx1100 "$python_bin" \
  >"$run_dir/hardware.json"
vulkaninfo --summary >"$run_dir/vulkaninfo_summary.txt" 2>&1
grep -q 'vendorID.*0x1002' "$run_dir/vulkaninfo_summary.txt"
grep -q 'RADV NAVI31' "$run_dir/vulkaninfo_summary.txt"

printf '%s\n' \
  'schema_version: radeon_oneloop.radeon_c_object_vksplat_preflight_run.v1' \
  'formal: false' \
  'host_role: radeon_c_formal_candidate_preflight' \
  'gpu: GPU0' \
  'gfx_target: gfx1100' \
  'vksplat_commit: e26c254938c81ff85998cd357a9e005e255d9b03' \
  'vksplat_checkout: clean_upstream' \
  'generated_fill_enabled: false' \
  'heldout_quality_claim: false' \
  >"$run_dir/manifest.yaml"

export PYTHONPATH="$repo_root/src:$repo_root"
export XDG_RUNTIME_DIR="$run_dir/xdg-runtime"
mkdir "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
started_ns=$(date +%s%N)
timeout --signal=TERM --kill-after=15 300 \
  "$python_bin" -m gaussian.vksplat_render_ply \
  --ply "$ply" \
  --cameras "$cameras" \
  --source-provenance "$provenance" \
  --output "$run_dir/render" \
  --vksplat-root "$vksplat_root" \
  --vksplat-commit e26c254938c81ff85998cd357a9e005e255d9b03 \
  >"$run_dir/render.log" 2>&1
finished_ns=$(date +%s%N)

"$python_bin" - "$run_dir/render/render_manifest.json" "$run_dir/metrics.json" \
  "$started_ns" "$finished_ns" <<'PY'
import json, sys
from pathlib import Path

render_path, output_path = map(Path, sys.argv[1:3])
started_ns, finished_ns = map(int, sys.argv[3:5])
render = json.loads(render_path.read_text(encoding="utf-8"))
checks = {
    "formal_flag_is_false_for_preflight": render["formal"] is False,
    "clean_pinned_vksplat_commit": render["vksplat_commit"]
    == "e26c254938c81ff85998cd357a9e005e255d9b03",
    "observed_ply_hash": render["ply_sha256"]
    == "ad30184dd9b39fdc211404538e70c0544d2c418b1af2003ae6f07353c5471183",
    "camera_hash": render["cameras_sha256"]
    == "4d03ed63829d786325adf29e4a6c1fb2542d0a090ba3b66ea1e8b099d994cb00",
    "provenance_hash": render["source_provenance_sha256"]
    == "9e8e70efb8c9d5d98e9e792f9f0c06836eb1127cdfd1e8d8dceda9e728441bf9",
    "gaussian_count_30000": render["gaussian_count"] == 30000,
    "four_nonempty_renders": len(render["renders"]) == 4
    and all(0.0 < item["mean_transmittance"] < 1.0 for item in render["renders"]),
    "generated_fill_ineligible": render["eligible_for_formal_metrics"] is False,
}
report = {
    "schema_version": "radeon_oneloop.radeon_c_object_vksplat_preflight.v1",
    "formal": False,
    "accepted": all(checks.values()),
    "checks": checks,
    "elapsed_s": (finished_ns - started_ns) / 1_000_000_000,
    "vram_bytes": render["vram_bytes"],
    "peak_vram_bytes": render["peak_vram_bytes"],
    "renders": render["renders"],
    "eligible_for_formal_registration": all(checks.values()),
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["accepted"]:
    raise RuntimeError("Radeon-c object VkSplat preflight failed")
PY

write_hashes
(cd "$run_dir" && sha256sum -c hashes.sha256 >/dev/null)
printf '{"status":"done_nonformal_formal_candidate_preflight"}\n' >"$run_dir/DONE"
trap - EXIT
printf 'Radeon-c object VkSplat preflight passed: %s\n' "$run_dir"
