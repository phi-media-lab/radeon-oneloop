#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  printf 'usage: %s <four-view-input-root> <pipeline-run-root> <seva-root> <local-model-root> <model-install-run-root> <python-bin>\n' "$0" >&2
  exit 64
fi

four_view=$1
pipeline_root=$2
seva_root=$3
model_root=$4
model_install_root=$5
python_bin=$6
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
pipeline_id="seva_primary_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
pipeline_dir="$pipeline_root/$pipeline_id"
generation_root="$pipeline_dir/generation"
audit_root="$pipeline_dir/audit"
generation_id="seva_four_view_${pipeline_id}_seed10027"
audit_id="seva_orbit_audit_${pipeline_id}"
generation_dir="$generation_root/$generation_id"
audit_dir="$audit_root/$audit_id"

[[ -d "$four_view" && -f "$four_view/DONE" ]]
[[ -d "$seva_root/.git" && -x "$python_bin" ]]
mkdir -p "$pipeline_root"
[[ ! -e "$pipeline_dir" ]]
mkdir -p "$pipeline_dir"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$pipeline_dir/DONE" ]]; then
    "$python_bin" - "$pipeline_dir/FAILED" "$status" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "radeon_oneloop.seva_primary_until_review_failure.v1",
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

if [[ ! -s "$model_root/modelv1.1.safetensors" || ! -f "$model_root/config.yaml" ]]; then
  "$runner_dir/install_phi_seva_model.sh" \
    "$model_root" "$model_install_root" "$python_bin" \
    >"$pipeline_dir/model_install.log" 2>&1
fi

ONELOOP_SEVA_RUN_ID="$generation_id" \
  "$runner_dir/run_phi_seva_four_view.sh" \
  "$four_view" "$generation_root" "$seva_root" "$model_root" "$python_bin" \
  >"$pipeline_dir/generation.log" 2>&1
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
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
metrics = json.loads((audit / "metrics.json").read_text())
review = {
    "schema_version": "radeon_oneloop.seva_review_required.v1",
    "formal": False,
    "status": "human_identity_temporal_review_required",
    "four_view_input": str(four_view),
    "generation_run": str(generation),
    "audit_run": str(audit),
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
    "credential_material_recorded": False,
}
(pipeline / "REVIEW_REQUIRED.json").write_text(
    json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
for name in ("generation.log", "audit.log", "REVIEW_REQUIRED.json"):
    if not (pipeline / name).is_file():
        raise SystemExit(f"pipeline evidence is missing: {name}")
lines = [
    f"{sha(pipeline / name)}  {name}"
    for name in ("generation.log", "audit.log", "REVIEW_REQUIRED.json")
]
if (pipeline / "model_install.log").is_file():
    lines.append(f"{sha(pipeline / 'model_install.log')}  model_install.log")
(pipeline / "hashes.sha256").write_text("\n".join(lines) + "\n")
done = {
    "schema_version": "radeon_oneloop.seva_primary_until_review_done.v1",
    "formal": False,
    "status": "done_pending_human_review",
    "review_required_sha256": sha(pipeline / "REVIEW_REQUIRED.json"),
    "hashes_sha256": sha(pipeline / "hashes.sha256"),
}
(pipeline / "DONE").write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
printf 'SEVA primary generation and numeric audit complete; human review required: %s\n' "$pipeline_dir"
