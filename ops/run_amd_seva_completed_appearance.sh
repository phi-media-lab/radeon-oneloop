#!/usr/bin/env bash
set -euo pipefail

# Negative-control compatibility shim.  This entry point used to fit SEVA
# colors while freezing the rejected four-mask visual-hull geometry.  It is
# intentionally non-runnable so that nobody can accidentally recreate the
# distorted asset that was visible in the old live demo.
printf '%s\n' \
  'REJECTED: the completed_appearance frozen-geometry branch is quarantined.' \
  'Use ops/run_amd_seva_full_geometry.sh; it rebuilds and optimizes geometry from the reviewed 49-view orbit.' \
  >&2
exit 65
