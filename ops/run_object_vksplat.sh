#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <dataset> <output-root> <python-bin> <vksplat-root> <repo-root> <steps> <smoke|sparse_static|mcmc>\n' "$0" >&2
  exit 64
}

[[ $# -eq 7 ]] || usage
dataset=$1
output_root=$2
python_bin=$3
vksplat_root=$4
repo_root=$5
steps=$6
mode=$7
trainer="$repo_root/gaussian/vksplat_train.py"

[[ -d "$dataset" ]]
[[ -f "$dataset/DONE" ]]
[[ -x "$python_bin" ]]
[[ -f "$vksplat_root/vksplat/simple_trainer.py" ]]
[[ -f "$trainer" ]]
[[ "$steps" =~ ^[1-9][0-9]*$ ]]
[[ "$mode" == smoke || "$mode" == sparse_static || "$mode" == mcmc ]] || usage

script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
run_id="vksplat_object_${mode}_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$output_root/$run_id"
train_dir="$run_dir/train"
mkdir -p "$run_dir"

cleanup() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    "$python_bin" - "$run_dir/FAILED" "$status" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status = sys.argv[1:]
value = {
    "stage": "radeon_f_nonformal_object_VkSplat",
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

vulkan_summary=$(vulkaninfo --summary 2>/dev/null)
if ! grep -q 'vendorID.*0x1002' <<<"$vulkan_summary" || ! grep -q 'RADV NAVI31' <<<"$vulkan_summary"; then
  printf '%s\n' 'expected a RADV NAVI31 Radeon device' >&2
  exit 69
fi
printf '%s\n' "$vulkan_summary" >"$run_dir/vulkaninfo_summary.txt"
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"

started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)
command=(
  "$python_bin" "$trainer"
  --source "$vksplat_root"
  --dataset "$dataset"
  --output "$train_dir"
  --image-dir images
  --mask-dir masks
  --sparse-dir sparse/0
  --min-images 4
  --steps "$steps"
  --eval-interval 8
  --host-role radeon_f_gpu0_gfx1100_nonformal
)
case "$mode" in
  smoke)
    command+=(--strategy default)
    ;;
  sparse_static)
    command+=(--strategy default --freeze-higher-sh --disable-refinement)
    ;;
  mcmc)
    command+=(--strategy mcmc --freeze-higher-sh --scale-reg 0.01 --opacity-reg 0.01)
    ;;
esac
XDG_RUNTIME_DIR="$xdg_runtime" timeout --signal=TERM --kill-after=30 1800 \
  "${command[@]}" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
finished_epoch=$(date +%s)
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

"$python_bin" - \
  "$run_dir/manifest.json" "$run_id" "$dataset" "$script_path" "$trainer" \
  "$started_utc" "$finished_utc" "$((finished_epoch - started_epoch))" \
  "$steps" "$mode" "$train_dir/oneloop_metrics.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
(
    manifest_path, run_id, dataset, runner, trainer, started_utc,
    finished_utc, runtime_s, steps, mode, metrics_path,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

metrics = json.loads(Path(metrics_path).read_text())
if metrics.get("formal") is not False:
    raise SystemExit("radeon-f metrics must be nonformal")
dataset_manifest = Path(dataset) / "dataset_manifest.json"
value = {
    "schema_version": "radeon_oneloop.object_vksplat_run.v1",
    "run_id": run_id,
    "formal": False,
    "host_role": "radeon_f_gpu0_gfx1100_nonformal",
    "mode": mode,
    "steps": int(steps),
    "started_utc": started_utc,
    "finished_utc": finished_utc,
    "runtime_s": int(runtime_s),
    "dataset_manifest_sha256": sha256(dataset_manifest),
    "runner_sha256": sha256(runner),
    "trainer_sha256": sha256(trainer),
    "metrics_sha256": sha256(metrics_path),
    "splat_sha256": sha256(Path(metrics_path).parent / "splat.ply"),
    "evaluation_present": metrics.get("evaluation") is not None,
    "acceptance_status": "smoke_only_not_an_asset" if mode == "smoke" else "pending_heldout_visual_and_metric_review",
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
    "stage": "radeon_f_nonformal_object_VkSplat",
    "status": "done_candidate",
    "manifest_sha256": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
open(done_path, "w", encoding="utf-8").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'Radeon-f object VkSplat %s complete: %s\n' "$mode" "$run_dir"
