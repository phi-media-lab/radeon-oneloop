#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  printf 'usage: %s <until-review-pipeline> <accepted-review.json> <observed-initialization> <output-root> <python-bin> [recovered-four-view-input]\n' "$0" >&2
  exit 64
fi

pipeline=$1
review=$2
observed_initialization=$3
output_root=$4
python_bin=$5
recovered_four_view=${6:-}
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[[ -d "$pipeline" && -f "$pipeline/DONE" && -f "$pipeline/REVIEW_REQUIRED.json" ]]
[[ -f "$review" && -d "$observed_initialization" && -x "$python_bin" ]]

mapfile -t bound_paths < <("$python_bin" - "$pipeline" "$review" "$recovered_four_view" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

pipeline = Path(sys.argv[1]).resolve()
review_path = Path(sys.argv[2]).resolve()
recovered_four_view = sys.argv[3]
request = json.loads((pipeline / "REVIEW_REQUIRED.json").read_text())
review = json.loads(review_path.read_text())
if request.get("schema_version") != "radeon_oneloop.seva_review_required.v1":
    raise SystemExit("unexpected SEVA review-request schema")
if request.get("automatic_promotion") is not False:
    raise SystemExit("SEVA pipeline weakened the human-review boundary")
if review.get("schema_version") != "radeon_oneloop.seva_four_view_orbit_review.v1":
    raise SystemExit("unexpected SEVA review schema")
if review.get("decision") != "accepted_low_confidence_pseudoviews":
    raise SystemExit("SEVA orbit was not accepted for pseudo-view training")
if all(name in request for name in ("generation_run", "audit_run", "four_view_input")):
    generation = Path(request["generation_run"]).resolve()
    audit = Path(request["audit_run"]).resolve()
    four_view = Path(request["four_view_input"]).resolve()
elif all(
    name in request
    for name in ("generation_run_id", "audit_run_id", "four_view_input_id")
):
    if not recovered_four_view:
        raise SystemExit("recovered SEVA review requires its explicit four-view input")
    generation = (pipeline / "generation" / request["generation_run_id"]).resolve()
    audit = (pipeline / "audit" / request["audit_run_id"]).resolve()
    four_view = Path(recovered_four_view).resolve()
    if generation.name != request["generation_run_id"]:
        raise SystemExit("recovered SEVA generation ID mismatch")
    if audit.name != request["audit_run_id"]:
        raise SystemExit("recovered SEVA audit ID mismatch")
    if four_view.name != request["four_view_input_id"]:
        raise SystemExit("recovered SEVA four-view input ID mismatch")
else:
    raise SystemExit("SEVA review request has no supported path binding")
if not generation.is_relative_to(pipeline / "generation"):
    raise SystemExit("generation run is outside the reviewed pipeline")
if not audit.is_relative_to(pipeline / "audit"):
    raise SystemExit("audit run is outside the reviewed pipeline")
audit_sha = hashlib.sha256((audit / "metrics.json").read_bytes()).hexdigest()
if request.get("audit_metrics_sha256") != audit_sha:
    raise SystemExit("review request no longer binds the selected audit")
if review.get("evidence", {}).get("audit_metrics_sha256") != audit_sha:
    raise SystemExit("accepted review does not bind the selected audit")
four_view_sha = hashlib.sha256((four_view / "manifest.json").read_bytes()).hexdigest()
if review.get("evidence", {}).get("four_view_manifest_sha256") != four_view_sha:
    raise SystemExit("accepted review does not bind the selected four-view input")
for path in (four_view, generation, audit):
    if not path.is_dir():
        raise SystemExit(f"reviewed SEVA input is missing: {path}")
    print(path)
PY
)
[[ ${#bound_paths[@]} -eq 3 ]]
four_view=${bound_paths[0]}
generation=${bound_paths[1]}
audit=${bound_paths[2]}

exec "$runner_dir/run_phi_seva_pseudoview_colmap.sh" \
  "$four_view" "$generation" "$audit" "$review" \
  "$observed_initialization" "$output_root" "$python_bin"
