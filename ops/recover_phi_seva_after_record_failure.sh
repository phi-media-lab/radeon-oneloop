#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  printf 'usage: %s <failed-pipeline-root> <pipeline-run-root> <four-view-input-root> <seva-root> <local-model-root> <runtime-s> <python-bin>\n' "$0" >&2
  exit 64
fi

failed_pipeline=$1
pipeline_root=$2
four_view=$3
seva_root=$4
model_root=$5
runtime_s=$6
python_bin=$7
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
pipeline_id="seva_primary_recovered_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
pipeline_dir="$pipeline_root/$pipeline_id"
generation_root="$pipeline_dir/generation"
audit_root="$pipeline_dir/audit"
generation_id="seva_four_view_${pipeline_id}_seed10027"
audit_id="seva_orbit_audit_${pipeline_id}"
generation_dir="$generation_root/$generation_id"
audit_dir="$audit_root/$audit_id"

[[ -d "$failed_pipeline" && -f "$failed_pipeline/FAILED" ]]
[[ ! -e "$failed_pipeline/DONE" ]]
[[ -d "$four_view" && -f "$four_view/DONE" ]]
[[ -d "$seva_root/.git" && -x "$python_bin" ]]
[[ -s "$model_root/modelv1.1.safetensors" && -f "$model_root/config.yaml" ]]
[[ "$runtime_s" =~ ^[0-9]+([.][0-9]+)?$ ]]
mapfile -t source_generations < <(
  find "$failed_pipeline/generation" -mindepth 1 -maxdepth 1 -type d -print
)
[[ ${#source_generations[@]} -eq 1 ]]
source_generation=${source_generations[0]}
[[ -f "$source_generation/FAILED" && ! -e "$source_generation/DONE" ]]
[[ -s "$source_generation/inference/samples-rgb.mp4" ]]
[[ -f "$source_generation/inference/transforms.json" ]]
[[ $(find "$source_generation/inference/samples-rgb" -maxdepth 1 -type f -name '*.png' | wc -l) -eq 49 ]]

mkdir -p "$pipeline_root"
[[ ! -e "$pipeline_dir" ]]
mkdir -p "$generation_dir" "$audit_root"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$pipeline_dir/DONE" ]]; then
    "$python_bin" - "$pipeline_dir/FAILED" "$status" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "radeon_oneloop.seva_primary_recovery_failure.v1",
    "formal": False,
    "status": "failed",
    "exit_code": int(sys.argv[2]),
    "credential_material_recorded": False,
}, indent=2, sort_keys=True) + "\n")
PY
    (
      cd "$pipeline_dir"
      find . -type f ! -name hashes.sha256 -print0 | sort -z | xargs -0 sha256sum >hashes.sha256
    )
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cp -a "$source_generation/inference" "$generation_dir/inference"
for name in command.sh environment.txt seva_commit.txt seva_git_status.txt \
  seva_patch.diff vae_provenance.json stdout.log stderr.log; do
  [[ -f "$source_generation/$name" ]]
  cp -a "$source_generation/$name" "$generation_dir/$name"
done

(
  cd "$source_generation/inference"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$generation_dir/source_inference.sha256"

"$python_bin" - "$source_generation" "$generation_dir" "$runtime_s" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

source, target = map(Path, sys.argv[1:3])
runtime_s = float(sys.argv[3])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
value = {
    "schema_version": "radeon_oneloop.seva_generation_recovery_parent.v1",
    "created_utc": datetime.now(timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z"),
    "formal": False,
    "recovery_reason": "completed_inference_rejected_only_by_3x4_camera_record_contract",
    "source_generation_id": source.name,
    "source_failed_sha256": sha(source / "FAILED"),
    "source_hashes_sha256": sha(source / "hashes.sha256"),
    "source_inference_index_sha256": sha(target / "source_inference.sha256"),
    "inference_rerun": False,
    "runtime_s": runtime_s,
    "credential_material_recorded": False,
}
(target / "recovery_parent.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

cd "$repo_root"
PYTHONPATH="$repo_root:$seva_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m gaussian.record_seva_four_view_run \
  --run-dir "$generation_dir" \
  --input-root "$four_view" \
  --seva-root "$seva_root" \
  --local-model-root "$model_root" \
  --revision e538e251c1009e9a41cf8b7fee5f21332a1960de \
  --seed 10027 \
  --runtime-s "$runtime_s" \
  >>"$generation_dir/stdout.log" 2>>"$generation_dir/stderr.log"
[[ -f "$generation_dir/DONE" ]]

ONELOOP_SEVA_AUDIT_RUN_ID="$audit_id" \
  "$runner_dir/run_phi_seva_orbit_audit.sh" \
  "$generation_dir" "$four_view" "$audit_root" "$python_bin" \
  >"$pipeline_dir/audit.log" 2>&1
[[ -f "$audit_dir/DONE" ]]

"$python_bin" - "$pipeline_dir" "$four_view" "$generation_dir" "$audit_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

pipeline, four_view, generation, audit = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metrics = json.loads((audit / "metrics.json").read_text())
review = {
    "schema_version": "radeon_oneloop.seva_review_required.v1",
    "formal": False,
    "status": "human_identity_temporal_review_required",
    "four_view_input": str(four_view.resolve()),
    "generation_run": str(generation.resolve()),
    "audit_run": str(audit.resolve()),
    "four_view_input_id": four_view.name,
    "generation_run_id": generation.name,
    "audit_run_id": audit.name,
    "audit_metrics_sha256": sha(audit / "metrics.json"),
    "generated_contact_sha256": sha(audit / "generated_contact.png"),
    "anchor_comparison_sha256": sha(audit / "real_generated_difference_anchors.png"),
    "numeric_summary": {
        "real_anchor_silhouette_iou": metrics["real_anchor_silhouette_iou"],
        "adjacent_foreground_iou": metrics["adjacent_foreground_iou"],
        "cyclic_seam": metrics["cyclic_seam"],
        "foreground_stability": metrics["foreground_stability"],
    },
    "required_human_checks": [
        "single_object_all_views",
        "front_face_confined_to_front_hemisphere",
        "two_asymmetric_ears_correct_sides",
        "rear_strap_and_keyring_stable",
        "no_duplicate_limb_or_floating_surface",
        "four_real_anchor_identities_preserved",
        "adjacent_motion_smooth",
        "cyclic_seam_unobtrusive",
        "background_stable",
        "private_hil_rear_top_identity_consistent",
    ],
    "automatic_promotion": False,
    "recovered_completed_inference": True,
    "credential_material_recorded": False,
}
(pipeline / "REVIEW_REQUIRED.json").write_text(
    json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
lines = [
    f"{sha(pipeline / name)}  {name}"
    for name in ("audit.log", "REVIEW_REQUIRED.json")
]
(pipeline / "hashes.sha256").write_text("\n".join(lines) + "\n")
done = {
    "schema_version": "radeon_oneloop.seva_primary_recovery_done.v1",
    "formal": False,
    "status": "done_pending_human_review",
    "inference_rerun": False,
    "review_required_sha256": sha(pipeline / "REVIEW_REQUIRED.json"),
    "hashes_sha256": sha(pipeline / "hashes.sha256"),
}
(pipeline / "DONE").write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'SEVA completed inference recovered and audited; human review required: %s\n' "$pipeline_dir"
